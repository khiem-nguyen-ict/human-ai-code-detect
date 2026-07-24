import logging
import os
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from src.models.evaluate import evaluate_model

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


class ClassificationHead(nn.Module):
    def __init__(self, base_model: nn.Module, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)
        return logits


class CodeDataset(Dataset):
    def __init__(self, input_ids: np.ndarray, attention_mask: np.ndarray, labels: np.ndarray):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def load_config() -> dict:
    import yaml

    config_path = Path("./config/config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    gradient_accumulation_steps: int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1

    return total_loss / num_batches


def eval_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0

    return {"loss": avg_loss, "accuracy": accuracy}


def save_reference_embeddings(
    base_model: nn.Module,
    tokenizer: AutoTokenizer,
    config: dict,
) -> None:
    processed_dir = config["data"]["processed_dir"]
    reference_path = config.get("monitoring", {}).get(
        "reference_data_path", "./data/processed/reference_embeddings.npy"
    )

    human_path = Path(processed_dir) / "human_tokenized.npz"
    ai_path = Path(processed_dir) / "ai_tokenized.npz"

    if not human_path.exists() or not ai_path.exists():
        logger.warning(
            "Tokenized data not found. Skipping reference embedding save."
        )
        return

    human_data = np.load(str(human_path), allow_pickle=True)
    ai_data = np.load(str(ai_path), allow_pickle=True)

    input_ids = np.concatenate(
        [human_data["input_ids"], ai_data["input_ids"]], axis=0
    )
    attention_mask = np.concatenate(
        [human_data["attention_mask"], ai_data["attention_mask"]], axis=0
    )

    dataset = CodeDataset(input_ids, attention_mask, np.zeros(len(input_ids)))
    dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    base_model.eval()
    embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids_batch = batch["input_ids"].to(base_model.device)
            attention_mask_batch = batch["attention_mask"].to(base_model.device)
            outputs = base_model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)

    embeddings = np.concatenate(embeddings, axis=0)
    out = Path(reference_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, embeddings)
    logger.info("Reference embeddings saved to %s, shape=%s", out, embeddings.shape)


def train() -> None:
    config = load_config()
    device = torch.device(os.getenv("DEVICE", "cpu"))

    processed_dir = config["data"]["processed_dir"]
    checkpoint_dir = config["model"]["checkpoint_dir"]
    onnx_dir = config["model"]["onnx_dir"]
    tokenizer_dir = config["model"]["tokenizer_dir"]

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(onnx_dir).mkdir(parents=True, exist_ok=True)
    Path(tokenizer_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model"])
    tokenizer.save_pretrained(tokenizer_dir)
    logger.info("Tokenizer saved to %s", tokenizer_dir)

    human_path = Path(processed_dir) / "human_tokenized.npz"
    ai_path = Path(processed_dir) / "ai_tokenized.npz"

    if not human_path.exists() or not ai_path.exists():
        raise FileNotFoundError(
            "Tokenized data not found. Run feature engineering first: "
            "python -m src.features.build_features"
        )

    human_data = np.load(str(human_path), allow_pickle=True)
    ai_data = np.load(str(ai_path), allow_pickle=True)

    input_ids = np.concatenate(
        [human_data["input_ids"], ai_data["input_ids"]], axis=0
    )
    attention_mask = np.concatenate(
        [human_data["attention_mask"], ai_data["attention_mask"]], axis=0
    )
    labels = np.concatenate(
        [human_data["labels"], ai_data["labels"]], axis=0
    )

    test_split = config["evaluation"]["test_split"]
    val_split = config["evaluation"]["val_split"]
    seed = config["project"]["seed"]

    train_val_input_ids, test_input_ids, train_val_attention_mask, test_attention_mask, train_val_labels, test_labels = train_test_split(
        input_ids,
        attention_mask,
        labels,
        test_size=test_split,
        random_state=seed,
        stratify=labels,
    )

    remaining = max(1.0 - test_split, 1e-9)
    val_frac = val_split / remaining

    train_input_ids, val_input_ids, train_attention_mask, val_attention_mask, train_labels, val_labels = train_test_split(
        train_val_input_ids,
        train_val_attention_mask,
        train_val_labels,
        test_size=val_frac,
        random_state=seed,
        stratify=train_val_labels,
    )

    train_dataset = CodeDataset(train_input_ids, train_attention_mask, train_labels)
    val_dataset = CodeDataset(val_input_ids, val_attention_mask, val_labels)
    test_dataset = CodeDataset(test_input_ids, test_attention_mask, test_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )

    logger.info(
        "Dataset split - train: %d, val: %d, test: %d",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    base_model = AutoModel.from_pretrained(config["model"]["base_model"])
    if config["training"]["freeze_encoder"]:
        for param in base_model.parameters():
            param.requires_grad = False

    model = ClassificationHead(base_model, num_labels=config["model"]["num_labels"])
    model.to(device)

    best_val_f1 = 0.0
    best_state = None

    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    total_steps = len(train_loader) * config["training"]["num_epochs"]
    warmup_steps = config["training"]["warmup_steps"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    for epoch in range(config["training"]["num_epochs"]):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            config["training"]["gradient_accumulation_steps"],
        )

        val_results = evaluate_model(model, val_loader, device)
        val_metrics = val_results["metrics"]
        val_f1 = val_metrics.get("f1", 0.0)
        val_acc = val_metrics.get("accuracy", 0.0)
        val_prec = val_metrics.get("precision", 0.0)
        val_rec = val_metrics.get("recall", 0.0)
        val_loss = val_metrics.get("loss", 0.0)

        logger.info(
            "Epoch %d/%d - Train Loss: %.4f - Val Loss: %.4f - Val Acc: %.4f - Val F1: %.4f - Val Prec: %.4f - Val Rec: %.4f",
            epoch + 1,
            config["training"]["num_epochs"],
            train_loss,
            val_loss,
            val_acc,
            val_f1,
            val_prec,
            val_rec,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_path = Path(checkpoint_dir) / "graphcodebert_best.pt"
            torch.save(best_state, best_path)
            logger.info("Best checkpoint saved to %s (val_f1=%.4f)", best_path, best_val_f1)

        checkpoint_path = Path(checkpoint_dir) / f"graphcodebert_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        logger.info("Checkpoint saved to %s", checkpoint_path)

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    final_path = Path(checkpoint_dir) / "graphcodebert_human_ai.pt"
    torch.save(model.state_dict(), final_path)
    logger.info("Final model saved to %s", final_path)

    test_results = evaluate_model(model, test_loader, device)
    test_metrics = test_results["metrics"]
    logger.info(
        "Test metrics - Acc: %.4f - F1: %.4f - Prec: %.4f - Rec: %.4f - AUC: %.4f",
        test_metrics.get("accuracy", 0.0),
        test_metrics.get("f1", 0.0),
        test_metrics.get("precision", 0.0),
        test_metrics.get("recall", 0.0),
        test_metrics.get("auc_roc", 0.0),
    )

    out = Path("./data/processed/test_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(out, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info("Test metrics saved to %s", out)

    save_reference_embeddings(base_model, tokenizer, config)

    if config["training"]["fp16"]:
        export_to_onnx(model, tokenizer, config)

    return model


def export_to_onnx(model: nn.Module, tokenizer: AutoTokenizer, config: dict) -> None:
    onnx_path = Path(config["model"]["onnx_fp16_path"])

    dummy_input_ids = torch.ones(1, config["model"]["max_seq_length"], dtype=torch.long)
    dummy_attention_mask = torch.ones(1, config["model"]["max_seq_length"], dtype=torch.long)

    model.eval()
    model.base_model.eval()

    fp32_path = onnx_path.with_suffix(".onnx")

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_length"},
            "attention_mask": {0: "batch_size", 1: "seq_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )

    logger.info("ONNX FP32 model exported to %s", fp32_path)

    import onnx

    model_fp32 = onnx.load(str(fp32_path))
    onnx.save(model_fp32, str(fp32_path))

    from onnx import TensorProto

    for initializer in model_fp32.graph.initializer:
        if initializer.data_type == TensorProto.FLOAT:
            initializer.raw_data = (
                np.frombuffer(initializer.raw_data, dtype=np.float32)
                .astype(np.float16)
                .tobytes()
            )
            initializer.data_type = TensorProto.FLOAT16

    for node in model_fp32.graph.node:
        for attr in node.attribute:
            if attr.HasField("f"):
                attr.f = np.float16(attr.f).item()

    onnx.save(model_fp32, str(onnx_path))
    logger.info("ONNX FP16 model saved to %s", onnx_path)

    if fp32_path.exists():
        fp32_path.unlink()


def main() -> None:
    train()


if __name__ == "__main__":
    main()
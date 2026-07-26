import logging
import os
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoModel

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


class CodeDataset(torch.utils.data.Dataset):
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


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    num_batches = 0
    criterion = CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            num_batches += 1

            probs = torch.softmax(logits, dim=-1)

            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="weighted"),
        "precision": precision_score(all_labels, all_preds, average="weighted"),
        "recall": recall_score(all_labels, all_preds, average="weighted"),
    }

    if len(np.unique(all_labels)) > 1:
        metrics["auc_roc"] = roc_auc_score(
            all_labels, all_probs[:, 1], multi_class="ovr"
        )
    else:
        metrics["auc_roc"] = 0.0

    report = classification_report(
        all_labels, all_preds, target_names=["human", "ai"], output_dict=True
    )

    logger.info("Evaluation metrics: %s", metrics)
    report_text = classification_report(
        all_labels, all_preds, target_names=["human", "ai"]
    )
    logger.info("Classification report:\n%s", report_text)

    return {
        "metrics": metrics,
        "report": report,
        "predictions": all_preds.tolist(),
        "probabilities": all_probs.tolist(),
        "labels": all_labels.tolist(),
    }


def main() -> None:
    device = torch.device(os.getenv("DEVICE", "cpu"))
    checkpoint_path = os.getenv("PYTORCH_MODEL_PATH", "./models/checkpoints/graphcodebert_human_ai.pt")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    base_model = AutoModel.from_pretrained("microsoft/graphcodebert-base")

    from src.models.train import ClassificationHead

    model = ClassificationHead(base_model, num_labels=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    from src.features.build_features import load_tokenized_data

    human_data = load_tokenized_data("human")
    ai_data = load_tokenized_data("ai")

    input_ids = np.concatenate([human_data["input_ids"], ai_data["input_ids"]], axis=0)
    attention_mask = np.concatenate([human_data["attention_mask"], ai_data["attention_mask"]], axis=0)
    labels = np.concatenate([human_data["labels"], ai_data["labels"]], axis=0)

    test_dataset = CodeDataset(input_ids, attention_mask, labels)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    results = evaluate_model(model, test_loader, device)

    out = Path("./data/processed/test_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(out, "w") as f:
        json.dump(results["metrics"], f, indent=2)

    logger.info("Evaluation complete. Results saved to %s", out)


if __name__ == "__main__":
    main()
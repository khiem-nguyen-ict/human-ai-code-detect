from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


class Predictor:
    def __init__(
        self,
        onnx_path: str | None = None,
        tokenizer_path: str | None = None,
    ):
        self.onnx_path = onnx_path or os.getenv(
            "ONNX_MODEL_PATH", "./models/onnx/graphcodebert_human_ai_fp16.onnx"
        )
        self.tokenizer_path = tokenizer_path or os.getenv(
            "TOKENIZER_PATH", "./tokenizer"
        )
        self.max_seq_length = int(os.getenv("MAX_SEQ_LENGTH", "512"))

        if not Path(self.onnx_path).exists():
            raise FileNotFoundError(f"ONNX model not found at {self.onnx_path}")

        self.session = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        logger.info("Predictor initialized with model at %s", self.onnx_path)

    def predict(self, code: str) -> dict:
        encoded = self.tokenizer(
            code,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="np",
        )

        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)

        outputs = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )

        logits = outputs[0]
        probabilities = 1.0 / (1.0 + np.exp(-logits))

        human_prob = float(probabilities[0][0])
        ai_prob = float(probabilities[0][1])
        prediction = "human" if human_prob > ai_prob else "ai"

        return {
            "prediction": prediction,
            "human_probability": round(human_prob, 4),
            "ai_probability": round(ai_prob, 4),
        }

    def predict_batch(self, codes: list[str]) -> list[dict]:
        return [self.predict(code) for code in codes]


def main() -> None:
    predictor = Predictor()

    sample_human = r"""
#include <stdio.h>
int main() {
    int n;
    scanf("%d", &n);
    int sum = 0;
    for (int i = 1; i <= n; i++) {
        sum += i;
    }
    printf("%d\n", sum);
    return 0;
}
"""

    result = predictor.predict(sample_human)
    logger.info("Prediction result: %s", result)


if __name__ == "__main__":
    main()
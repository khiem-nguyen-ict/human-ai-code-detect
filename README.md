# Human vs AI Code Detection

Detect whether C source code is human-written or AI-generated using GraphCodeBERT fine-tuned with a linear classification head, exported to ONNX FP16 for offline inference.

## What it does

- Loads a labeled dataset of human-written and AI-generated C code.
- Fine-tunes `microsoft/graphcodebert-base` by freezing the encoder and training a 768 → 2 linear classifier.
- Serializes the fine-tuned model to PyTorch (`graphcodebert_human_ai.pt`).
- Exports to ONNX FP16 (`graphcodebert_human_ai_fp16.onnx`) for CPU-friendly inference.
- Runs predictions on raw C files or code strings, returning Human / AI probabilities.

## Prerequisites

- Python 3.9+
- pip
- Jupyter or JupyterLab

### Python dependencies

```bash
pip install torch transformers sentencepiece onnx onnxruntime onnxconverter-common
```

## Dataset

`dataset.zip` should contain:

```
dataset/
├── human/   # C files labeled as human-written (label 0)
└── ai/      # C files labeled as AI-generated  (label 1)
```

## Run

1. Extract the dataset:

```bash
python -c "import zipfile; zipfile.ZipFile('dataset.zip').extractall('.')"
```

2. Open the notebook:

```bash
jupyter lab GraphCodeBERT_Course_Detect_Human_and_AI_code.ipynb
# or
jupyter notebook GraphCodeBERT_Course_Detect_Human_and_AI_code.ipynb
```

3. Run cells in order. The notebook will:
   - Install dependencies
   - Tokenize and embed the C code with GraphCodeBERT
   - Train the classification head
   - Save `graphcodebert_human_ai.pt`
   - Export to `graphcodebert_human_ai.onnx`
   - Convert to FP16 and save `graphcodebert_human_ai_fp16.onnx`
   - Load the ONNX model and run predictions

## Model files

| File | Description |
|------|-------------|
| `graphcodebert_human_ai.pt` | Fine-tuned PyTorch checkpoint |
| `graphcodebert_human_ai.onnx` | ONNX FP32 model |
| `graphcodebert_human_ai_fp16.onnx` | ONNX FP16 model for production |

## Architecture

```
C source code
    ↓
Tokenizer (microsoft/graphcodebert-base)
    ↓
input_ids + attention_mask
    ↓
GraphCodeBERT encoder
    ↓
CLS embedding (768-d)
    ↓
Linear(768 → 2)
    ↓
Softmax
    ↓
Human / AI
```

## Predicted files

- `GraphCodeBERT_Course_Detect_Human_and_AI_code.ipynb` — main training and inference notebook
- `dataset.zip` — course dataset with human and AI C files

## Notes

- GraphCodeBERT is best suited for C/C++/Java/Python. This course exercises it on C code.
- The ONNX model is standalone for the neural network path. The tokenizer is still loaded from Microsoft for production use.
- For fully offline deployment, save the tokenizer locally with `tokenizer.save_pretrained("./tokenizer")`.

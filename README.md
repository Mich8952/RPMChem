# Fine-Tuning LLMs for Chemistry Problem Solving
### Created by the RPMChem team.

This is a research pipeline for fine-tuning lightweight LLMs on physical chemistry textbook Q&A data using QLoRA on Apple Silicon (MLX). The pipeline covers PDF extraction, dataset preprocessing, LoRA training, and evaluation.

> **Platform**: macOS (Apple Silicon, M-series) — the training stack uses [MLX](https://github.com/ml-explore/mlx) and is not compatible with CUDA/Linux out of the box.

***

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Fusing Weights](#fusing-weights)
- [Evaluation](#evaluation)
- [Configuration Reference](#configuration-reference)
- [Notes for Contributors](#notes-for-contributors)

***

## Project Overview

The pipeline fine-tunes a quantized Llama 3.1 8B model (or any HuggingFace-compatible model) on physical chemistry problem/solution pairs extracted from PDFs. Training uses QLoRA via MLX for memory-efficient training on Apple Silicon. Evaluation compares the fine-tuned model against the base model using SciBERT-based semantic scoring and numerical relative error.

**Pipeline stages:**

1. **Preprocessing** — Extract Q/A pairs from PDFs, optionally impute reasoning traces
2. **Training** — QLoRA fine-tuning via MLX
3. **Fusing** — Merge LoRA adapters into the base model weights
4. **Evaluation** — Compare models on semantic (SciBERTScore, ROUGE-L) and numerical (relative error) metrics

***

## Repository Structure

```
RPMChem/
├── conf/
│   ├── preprocessing.yaml       # Preprocessing config (PDF paths, modes)
│   └── training.yaml            # Training hyperparameters
├── preprocessing/
│   ├── preprocessor_pipeline.py # Main preprocessing entry point
│   ├── get_jsons_joint.py       # Extractor for joint Q/A PDFs
│   ├── get_jsons_disjoint_textbook.py  # Extractor for disjoint PDFs
│   ├── combine_jsons_disjoint.py
│   ├── combine_textbooks.py
│   ├── re_process_real.py       # Filters bad samples, splits train/valid
│   ├── add_reasoning_context.py # Optionally adds LLM reasoning traces
│   ├── extract_numerical_subset.py
│   ├── prompts.py               # All LLM prompt templates
│   └── mcmurray_extractor.py    # Organic chemistry extractor (OpenStax)
├── training/
│   ├── train.py                 # Main training entry point
│   ├── models.py                # Model loading, LoRA layer injection
│   ├── dataclasses_mlx.py       # Dataset/DataLoader for MLX
│   ├── early_stopper.py
│   └── tokenizer_template.py
├── analysis/
│   ├── run_test_semantics.py          # Semantic evaluation (BERTScore + ROUGE-L)
│   ├── run_test_numerical.py          # Numerical evaluation (relative error)
│   ├── run_test_semantics_PE_cot.py   # Semantic eval with chain-of-thought
│   ├── run_test_numerical_PE_cot.py   # Numerical eval with chain-of-thought
│   ├── run_stat_test_on_semantics.py  # Statistical significance tests
│   ├── run_stat_test_on_numerical.py
│   ├── stat_classes.py
│   └── results/                       # CSV outputs saved here
├── datasets/                    # Place your PDFs and generated datasets here
│   ├── raw/                     # Engel & Reid PDF
│   └── pdfs/                    # Atkins PDF(s)
└── fuse.md                      # Notes on fusing LoRA adapters
```

***

## Requirements

- **macOS with Apple Silicon** (M1/M2/M3/M4)
- Python 3.10+
- [LM Studio](https://lmstudio.ai/) — required only for the reasoning imputation step (`add_reasoning_context.py`), which calls a local model via the `lmstudio` Python SDK

### Python Dependencies

Install all dependencies from the project root:

```bash
pip install -r requirements.txt
```

> **Note**: A `requirements.txt` is not yet committed. Install the following manually until it is added:

```bash
pip install mlx mlx-lm transformers huggingface_hub \
            pymupdf pypdf pyyaml \
            bert-score rouge-score scikit-learn \
            pandas numpy matplotlib tqdm lmstudio orjsonl
```

***

## Setup

```bash
git clone https://github.com/Mich8952/RPMChem.git
cd RPMChem
```

All scripts are designed to be run from the **repository root**. Do not run scripts from inside a subdirectory — relative paths like `datasets/`, `conf/`, and `datasets/e2e_artifacts/` are resolved relative to the working directory.

```bash
# Correct
python preprocessing/preprocessor_pipeline.py --config conf/preprocessing.yaml

# Incorrect — will break relative paths
cd preprocessing
python preprocessor_pipeline.py
```

***

## Data Preparation

The pipeline requires physical chemistry textbook PDFs. These are copyrighted and cannot be redistributed with the repo.

[Textbook files can be found and downloaded here.](https://drive.google.com/drive/folders/1N8KDausUHZJRQ96z_Mpe4YoxXxWVbjsn?usp=sharing)

| File | Destination | Notes |
|---|---|---|
| Atkins' Physical Chemistry 11e | `datasets/pdfs/` | Main textbook (optional for preprocessing) |
| Solutions Manual — Atkins 11th Ed | `datasets/pdfs/` | Paired solutions |
| Engel & Reid Physical Chemistry Solution Manual | `datasets/raw/` | Used in disjoint mode |

Place the PDFs in the directories above before running preprocessing. Path names in `conf/preprocessing.yaml` should match exactly.

***

## Preprocessing

The preprocessing pipeline extracts Q/A pairs from PDFs, cleans them, splits into train/validation, and optionally augments with reasoning traces.

### Configure `conf/preprocessing.yaml`

```yaml
textbook_dir: datasets          # Base directory for PDF paths (relative to repo root)
textbooks:
  - mode: joint                 # Questions and solutions in the same PDF
    textbook_pdf: raw/BookA.pdf

  - mode: disjoint              # Questions and solutions in separate PDFs
    question_pdf: pdfs/BookB_Questions.pdf
    solutions_pdf: pdfs/BookB_Solutions.pdf
```

**Modes:**
- `joint` — Q and A are in the same PDF (e.g., a combined textbook+solutions)
- `disjoint` — Q and A are in separate PDFs (e.g., textbook + solutions manual)

**Path resolution:** Paths under `textbooks` are relative to `textbook_dir`. You can also use absolute paths.

### Run Preprocessing

```bash
# Full pipeline (with reasoning imputation — requires LM Studio running locally)
python preprocessing/preprocessor_pipeline.py --config conf/preprocessing.yaml

# Skip reasoning imputation
python preprocessing/preprocessor_pipeline.py --config conf/preprocessing.yaml --no-impute
```

**Outputs** are written to `datasets/e2e_artifacts/` with session-scoped filenames (UUID-based) to avoid collisions across runs. Final train/validation splits are named:

- `datasets/e2e_artifacts/train_<session>.jsonl`
- `datasets/e2e_artifacts/valid_<session>.jsonl`
- `datasets/e2e_artifacts/train_IMPUTED.jsonl` / `valid_IMPUTED.jsonl` (if imputation is run)

Copy or symlink the desired split into `datasets/current_to_run/` before training (or update `conf/training.yaml` with the exact path).

### Reasoning Imputation (Optional)

The `--no-impute` flag skips the `add_reasoning_context.py` step, which calls a locally running `gpt-oss-120b` model via LM Studio. If you do not have LM Studio set up, always use `--no-impute`.

To use imputation:
1. Download and install [LM Studio](https://lmstudio.ai/)
2. Load a model (e.g., `gpt-oss-120b` or equivalent)
3. Start the local server
4. Run without `--no-impute`

***

## Training

### Configure `conf/training.yaml`

```yaml
model_dir: mlx-community/Meta-Llama-3.1-8B-Instruct-4bit  # HF repo ID or local path
train_jsonl: datasets/current_to_run/train_IMPUTED.jsonl
valid_jsonl: datasets/current_to_run/valid_IMPUTED.jsonl   # or "split_from_train"
save_dir: adapters_v1
max_seq_len: 5000
batch_size: 1
iters: 10000
eval_every: 100
eval_batches: 125
save_every: 250
apply_chat_template: true
mask_prompt: true
system_prompt: ""
lr: 1.0e-5
weight_decay: 0.0
lora_rank: 16
lora_alpha: 32.0
lora_dropout: 0.0
num_layers: -1   # -1 = apply LoRA to all layers
seed: 42
```

**Key notes:**
- `model_dir` accepts a HuggingFace repo ID (auto-downloaded) or a local directory path
- Set `valid_jsonl: split_from_train` to auto-generate a 70/15/15 train/valid/test split from the training file
- `num_layers: -1` applies LoRA to all transformer layers
- Adapters and checkpoints are saved to `save_dir` under the repo root

### Run Training

```bash
# Run from repo root
python training/train.py --config conf/training.yaml
```

Training logs are written to `logs/run_<uuid>.log`. A live validation loss plot is saved to `train_curr_temp.png` in the repo root during training.

**Outputs:**
- `<save_dir>/adapters.safetensors` — latest adapter checkpoint
- `<save_dir>/lora_final.safetensors` — final or early-stopped adapter
- `<save_dir>/lora_step_XXXXXXX.safetensors` — periodic checkpoints
- `<save_dir>/adapter_config.json` — full training config snapshot
- `<save_dir>/results.pkl` — loss curves

***

## Fusing Weights

After training, merge the base model weights with the LoRA adapters to produce a single deployable model:

```bash
python -m mlx_lm.fuse \
  --model /path/to/base_model_dir \
  --adapter-path /path/to/adapter_dir \
  --save-path /path/to/fused_model_dir
```

Then synchronize the tokenizer and system prompt into the fused model directory:

```bash
python -m mlx_lm.fuse \
  --model ~/.cache/huggingface/hub/models--mlx-community--Meta-Llama-3.1-8B-Instruct-4bit/snapshots/<hash> \
  --adapter-path adapters_v1 \
  --save-path fused_model_v1
```

See `fuse.md` for additional notes on tokenizer syncing.

***

## Evaluation

All evaluation scripts are in `analysis/` and must be run from the **repo root**.

### Semantic Evaluation (BERTScore + ROUGE-L)

Compares two models on a validation set using SciBERT-based BERTScore and ROUGE-L:

```bash
python analysis/run_test_semantics.py \
  --dataset_dir datasets/current_to_run/valid_IMPUTED.jsonl \
  --model_1 ~/.lmstudio/models/personal/8b_nolora \
  --model_2 ~/.lmstudio/models/submission/fuse_model_8b_qlora_manual_NEW_prompt
```

Results are saved to `analysis/results/semantics_<timestamp>.csv`.

### Numerical Evaluation (Relative Error)

Compares models on a CSV of numerical Q/A pairs, extracting final answers and computing relative error:

```bash
python analysis/run_test_numerical.py \
  --dataset_dir datasets/numerical_prompts_real/validation.csv \
  --model_1 ~/.lmstudio/models/personal/8b_noLora \
  --model_2 ~/.lmstudio/models/submission/fuse_model_8b_qlora_manual_NEW_prompt
```

Results are saved to `analysis/results/numerical_<timestamp>.csv`.

### Chain-of-Thought Variants

`run_test_semantics_PE_cot.py` and `run_test_numerical_PE_cot.py` run the same evaluations but strip reasoning preamble from fine-tuned model outputs before scoring.

### Statistical Tests

```bash
python analysis/run_stat_test_on_semantics.py
python analysis/run_stat_test_on_numerical.py
```

***

## Configuration Reference

### `conf/preprocessing.yaml`

| Key | Description |
|---|---|
| `textbook_dir` | Base directory for all PDF paths (relative to repo root) |
| `textbooks` | List of textbook entries |
| `mode` | `joint` or `disjoint` |
| `textbook_pdf` | Path to joint PDF (relative to `textbook_dir`) |
| `question_pdf` | Path to questions PDF (disjoint mode) |
| `solutions_pdf` | Path to solutions PDF (disjoint mode) |

### `conf/training.yaml`

| Key | Description | Default |
|---|---|---|
| `model_dir` | HF repo ID or local model path | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` |
| `train_jsonl` | Path to training JSONL | — |
| `valid_jsonl` | Path to validation JSONL, or `split_from_train` | — |
| `save_dir` | Output directory for adapters | `adapters_vx` |
| `max_seq_len` | Max token length per sample | `5000` |
| `batch_size` | Training batch size | `1` |
| `iters` | Total training iterations | `10000` |
| `eval_every` | Evaluate every N steps | `100` |
| `eval_batches` | Batches to use per eval call | `125` |
| `save_every` | Save checkpoint every N steps | `250` |
| `apply_chat_template` | Apply tokenizer chat template | `true` |
| `mask_prompt` | Mask prompt tokens in loss | `true` |
| `system_prompt` | System prompt baked into adapter | `""` |
| `lr` | Learning rate | `1e-5` |
| `weight_decay` | AdamW weight decay | `0.0` |
| `lora_rank` | LoRA rank | `16` |
| `lora_alpha` | LoRA alpha (scaling) | `32.0` |
| `lora_dropout` | LoRA dropout | `0.0` |
| `num_layers` | Number of layers to apply LoRA (`-1` = all) | `-1` |
| `seed` | Random seed | `42` |

***

## Notes for Contributors

- **Always run scripts from the repo root.** All relative paths (`datasets/`, `conf/`, `logs/`, `analysis/results/`) are anchored to the root.
- **Model paths in eval scripts** default to `~/.lmstudio/models/...`. Override with `--model_1` / `--model_2` CLI args when running on a different machine.
- **`mcmurray_extractor.py`** is an older Colab-style notebook converted to a script. It uses `!pip install` shell commands and `fitz.open('textbook1.pdf')` with a hardcoded filename — it requires manual editing before use. A future refactor should convert it to accept a CLI argument for the PDF path.
- **`analysis/.DS_Store`** — this macOS metadata file should be added to `.gitignore` if not already excluded.
- The `datasets/` directory is gitignored (large/copyrighted files). Any new contributor must supply their own PDFs.

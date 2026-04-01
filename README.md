@ Rosie, please fill out. Please see the notes below for running


## Preprocessing
Use `conf/preprocessing.yaml` to define what gets processed.

`joint` means questions and solutions are in the same PDF.
`disjoint` means questions and solutions are in separate PDFs.

Minimal config format:

```yaml
textbook_dir: datasets
textbooks:
  - mode: joint
    textbook_pdf: raw/BookA.pdf
  - mode: disjoint
    question_pdf: pdfs/BookB_Questions.pdf
    solutions_pdf: pdfs/BookB_Solutions.pdf
```

Notes:
1. `textbook_dir` is the base directory for relative PDF paths (relative to where you run the script).
2. For `joint`, only set `textbook_pdf`.
3. For `disjoint`, set both `question_pdf` and `solutions_pdf`.
4. You can also use absolute paths instead of relative paths.
5. You may call scripts from the CLI as shown below or use your IDE. Argparsing defaults are set to use the config.

Run e2e:

```bash
python preprocessing/preprocessor_pipeline.py --config conf/preprocessing.yaml
```

Optional (skip reasoning imputation):

```bash
python preprocessing/preprocessor_pipeline.py --config conf/preprocessing.yaml --no-impute
```

## Training
Use `conf/training.yaml` to define training data paths and hyperparameters.

Minimal config format:

```yaml
model_dir: mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
train_jsonl: datasets/current_to_run/train_IMPUTED.jsonl
valid_jsonl: datasets/current_to_run/valid_IMPUTED.jsonl
save_dir: adapters_vx
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
num_layers: -1
seed: 42
```

Notes:
1. YAML keys map directly to `train(...)` args in `training/train.py`.
2. Set `valid_jsonl: split_from_train` to auto-create a validation split from the train file.
3. `model_dir` can be a local model path or a Hugging Face repo ID.

Run training:

```bash
python training/train.py --config conf/training.yaml
```

## Fuse Weights
After training, fuse base model + LoRA adapters:

```bash
python -m mlx_lm.fuse \
  --model /path/to/base_model_dir \
  --adapter-path /path/to/adapter_dir \
  --save-path /path/to/fused_model_dir
```

Then sync tokenizer/template into the fused model:

```bash
python /path/to/sync_fused_tokenizer.py \
  --source-model-dir /path/to/base_model_dir \
  --fused-model-dir /path/to/fused_model_dir \
  --system-prompt "$(python - <<'PY'
import json
p='/path/to/adapter_config.json'
print(json.load(open(p, encoding='utf-8'))['system_prompt'])
PY
)"
```

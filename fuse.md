
#FUSE
python -m mlx_lm.fuse \
  --model /Users/michaelmurray/.cache/huggingface/hub/models--mlx-community--Meta-Llama-3.1-8B-Instruct-4bit/snapshots/241a666dad6cb93c8ff213d39a7f34a36bf26db4 \
  --adapter-path /Users/michaelmurray/Documents/GitHub/RPMChem/adapters_manual \
  --save-path /Users/michaelmurray/.lmstudio/models/personal/fuse_model_8b_qlora_manual_NEW


#SYNC TOKENIZER
python /Users/michaelmurray/Documents/GitHub/RPMChem/mlx_manual/sync_fused_tokenizer.py \
  --source-model-dir /Users/michaelmurray/.cache/huggingface/hub/models--mlx-community--Meta-Llama-3.1-8B-Instruct-4bit/snapshots/241a666dad6cb93c8ff213d39a7f34a36bf26db4 \
  --fused-model-dir /Users/michaelmurray/.lmstudio/models/personal/fuse_model_8b_qlora_manual_NEW \
  --system-prompt "$(python - <<'PY'
import json
p='/Users/michaelmurray/Documents/GitHub/chem_llm/adapters_manual/adapter_config.json'
print(json.load(open(p, encoding='utf-8'))['system_prompt'])
PY
)"

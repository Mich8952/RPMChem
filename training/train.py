import argparse
import copy
import itertools
import json
import math  # noqa: F401  (kept for potential downstream use)
import os
import pickle
import shutil
from uuid import uuid4

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import yaml
from transformers import AutoTokenizer

from dataclasses_mlx import DataLoader, JSONLDataset
from early_stopper import EarlyStopper
from models import (
    causal_lm_loss,
    linear_to_lora_layers,
    load_pretrained_model,
    save_lora_adapters,
)
from tokenizer_template import patch_chat_template_jinja, patch_tokenizer_config

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

# Inspiration/help for some code was taken from the mlx-lm package.
# We wanted more controllability than mlx-lm exposes, and mlx is
# significantly faster than PyTorch MPS (which also lacks 4-bit training).


def resolve_model_dir(model_dir_or_repo):
    """Return a local model directory, downloading from HF Hub if needed."""
    if os.path.exists(model_dir_or_repo):
        return model_dir_or_repo
    local_dir = snapshot_download(
        repo_id=model_dir_or_repo,
        allow_patterns=[
            "config.json",
            "*.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ],
    )
    return local_dir


def get_context_limit(model_config, tokenizer):
    """Return the hard context window size from config or tokenizer."""
    max_pos = model_config.get("max_position_embeddings")
    if isinstance(max_pos, int):
        return max_pos
    tok_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_max, int):
        return tok_max
    return None


def convert_batch_to_dct(batch):
    return {
        "input_ids": mx.array(batch["input_ids"]),
        "labels": mx.array(batch["labels"]),
    }


def build_adapter_config(
    model_dir,
    train_jsonl,
    seed,
    num_layers,
    batch_size,
    iters,
    eval_batches,
    lr,
    eval_every,
    save_every,
    save_dir,
    max_seq_len,
    mask_prompt,
    system_prompt,
    lora_rank,
    lora_dropout,
    lora_alpha,
):
    """Serialize training hyperparameters to a dict for later JSON storage.

    eval_batches controls how many batches are used during each eval
    call (effective samples = batch_size * eval_batches).
    """
    data_dir = os.path.dirname(os.path.abspath(train_jsonl))
    return {
        "model": model_dir,
        "train": True,
        "fine_tune_type": "lora",
        "data": data_dir,
        "seed": seed,
        "num_layers": num_layers,
        "batch_size": batch_size,
        "iters": iters,
        "val_batches": eval_batches,
        "learning_rate": lr,
        "steps_per_report": 10,
        "steps_per_eval": eval_every,
        "save_every": save_every,
        "adapter_path": save_dir,
        "resume_adapter_file": None,
        "max_seq_length": max_seq_len,
        "grad_checkpoint": False,
        "grad_accumulation_steps": 1,
        "mask_prompt": mask_prompt,
        "system_prompt": system_prompt,
        "lora_parameters": {
            "rank": lora_rank,
            "dropout": lora_dropout,
            "scale": lora_alpha,
        },
    }


def evaluate(model, loader, max_batches):
    """Compute mean cross-entropy loss over up to max_batches batches."""
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        b = convert_batch_to_dct(batch)
        logits = model(b["input_ids"])
        loss = causal_lm_loss(logits, b["labels"])
        losses.append(float(loss.item()))

    if not losses:
        return float("nan")
    return sum(losses) / len(losses)


def copy_tokenizer_artifacts_from_orig_model(tokenizer_path, save_dir):
    """Copy tokenizer files from the base model into the adapter directory."""
    if not os.path.exists(tokenizer_path):
        return

    artifact_names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    )
    for name in artifact_names:
        source_file = os.path.join(tokenizer_path, name)
        target_file = os.path.join(save_dir, name)
        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)


def bake_prompt_into_saved_chat_template(save_dir, system_prompt):
    """Embed the system prompt into the saved tokenizer/chat template.

    Required so inference loads the correct prompt without extra config.
    """
    tokenizer_config_path = os.path.join(save_dir, "tokenizer_config.json")
    chat_template_path = os.path.join(save_dir, "chat_template.jinja")

    updated_config = patch_tokenizer_config(tokenizer_config_path, system_prompt)
    updated_jinja = patch_chat_template_jinja(chat_template_path, system_prompt)

    if updated_config:
        print(f"updated prompt: {tokenizer_config_path}")
    else:
        print(f"unchanged: {tokenizer_config_path}")

    if updated_jinja:
        print(f"updated prompt: {chat_template_path}")
    elif os.path.exists(chat_template_path):
        print(f"unchanged: {chat_template_path}")


def train(
    model_dir,
    train_jsonl,
    valid_jsonl,
    save_dir,
    max_seq_len,
    batch_size,
    iters,
    eval_every,
    eval_batches,
    save_every,
    apply_chat_template,
    mask_prompt,
    system_prompt,
    lr,
    weight_decay,
    lora_rank,
    lora_alpha,
    lora_dropout,
    num_layers,
    seed,
):
    """Train QLoRA adapters on a JSONL dataset.

    Follows a PyTorch-style training loop implemented in MLX.
    """
    mx.random.seed(seed)

    model_dir = resolve_model_dir(model_dir)
    tokenizer_path = model_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        model_config = json.load(f)

    # Build train/validation datasets
    if valid_jsonl == "split_from_train":
        # 17.65% of the 85% train portion gives a final 70/15/15 split
        split_prop = 0.1765
        train_ds = JSONLDataset(
            train_jsonl,
            tokenizer,
            max_length=max_seq_len,
            apply_chat_template=apply_chat_template,
            mask_prompt=mask_prompt,
            system_prompt=system_prompt,
            split_prop=split_prop,
            set_type="train",
        )
        valid_ds = JSONLDataset(
            train_jsonl,
            tokenizer,
            max_length=max_seq_len,
            apply_chat_template=apply_chat_template,
            mask_prompt=mask_prompt,
            system_prompt=system_prompt,
            split_prop=split_prop,
            set_type="valid",
        )
    else:
        train_ds = JSONLDataset(
            train_jsonl,
            tokenizer,
            max_length=max_seq_len,
            apply_chat_template=apply_chat_template,
            mask_prompt=mask_prompt,
            system_prompt=system_prompt,
        )
        valid_ds = JSONLDataset(
            valid_jsonl,
            tokenizer,
            max_length=max_seq_len,
            apply_chat_template=apply_chat_template,
            mask_prompt=mask_prompt,
            system_prompt=system_prompt,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        pad_token_id=tokenizer.pad_token_id,
        shuffle=True,
        seed=seed,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        pad_token_id=tokenizer.pad_token_id,
        shuffle=False,
        seed=seed,
    )

    model, _ = load_pretrained_model(model_dir)
    model.freeze()

    if num_layers > len(model.layers):
        raise ValueError(
            f"You asked for {num_layers} LoRA layers, but the model only "
            f"has {len(model.layers)}. Please reduce num_layers."
        )

    linear_to_lora_layers(
        model,
        num_layers=num_layers,
        rank=lora_rank,
        scale=lora_alpha,
        dropout=lora_dropout,
    )

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)

    def loss_fn(cur_model, cur_batch):
        logits = cur_model(cur_batch["input_ids"])
        return causal_lm_loss(logits, cur_batch["labels"])

    # value_and_grad computes loss + gradients in one pass (like
    # loss.backward() in PyTorch)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Initialise save directory and persist adapter config
    os.makedirs(save_dir, exist_ok=True)
    adapter_config_path = os.path.join(save_dir, "adapter_config.json")
    with open(adapter_config_path, "w", encoding="utf-8") as f:
        json.dump(
            build_adapter_config(
                model_dir=model_dir,
                train_jsonl=train_jsonl,
                seed=seed,
                num_layers=num_layers,
                batch_size=batch_size,
                iters=iters,
                eval_batches=eval_batches,
                lr=lr,
                eval_every=eval_every,
                save_every=save_every,
                save_dir=save_dir,
                max_seq_len=max_seq_len,
                mask_prompt=mask_prompt,
                system_prompt=system_prompt,
                lora_rank=lora_rank,
                lora_dropout=lora_dropout,
                lora_alpha=lora_alpha,
            ),
            f,
            indent=4,
        )
    print(f"saved={adapter_config_path}")

    prompt_config_path = os.path.join(save_dir, "chem_prompt_config.json")
    with open(prompt_config_path, "w", encoding="utf-8") as f:
        json.dump({"system_prompt": system_prompt}, f, indent=2)

    copy_tokenizer_artifacts_from_orig_model(tokenizer_path, save_dir)
    bake_prompt_into_saved_chat_template(save_dir, system_prompt)

    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"run_{uuid4()}.log")

    train_losses, valid_losses = [], []
    train_steps, valid_steps = [], []

    es = EarlyStopper(patience=5)
    train_iter = itertools.cycle(train_loader)

    with open(log_path, "a", encoding="utf-8") as log_file:
        for step in range(iters):
            batch = convert_batch_to_dct(next(train_iter))
            # loss_and_grad mirrors PyTorch's loss + loss.backward()
            loss, grads = loss_and_grad(model, batch)

            # optimizer.update plans the update; mx.eval materialises it
            optimizer.update(model, grads)
            mx.eval(loss, model.parameters(), optimizer.state)

            if step % 10 == 0 or step == 0:
                msg = f"step={step} train_loss={float(loss.item()):.6f}"
                print(msg)
                log_file.write(msg + "\n")
                log_file.flush()

            if step % 30 == 0:
                train_steps.append(step)
                train_losses.append(loss.item())

            run_eval = (step % eval_every == 0 and step != 0) or step == 2
            if run_eval:
                try:
                    val_loss = evaluate(model, valid_loader, eval_batches)
                    msg = f"step={step} val_loss={val_loss:.6f}"
                    print(msg)
                    log_file.write(msg + "\n")
                    log_file.flush()

                    valid_losses.append(val_loss)
                    valid_steps.append(step)

                    plt.clf()
                    plt.plot(valid_steps, valid_losses, label="val")
                    plt.legend()
                    plt.savefig("train_curr_temp")

                    stopped, best_model = es(val_loss, curr_model=copy.deepcopy(model))
                    if stopped:
                        print("Early stopping triggered")
                        final_ckpt = os.path.join(save_dir, "lora_final.safetensors")
                        save_lora_adapters(best_model, final_ckpt)
                        save_lora_adapters(
                            best_model,
                            os.path.join(save_dir, "adapters.safetensors"),
                        )
                        results = {
                            "train_loss": [float(v) for v in train_losses],
                            "valid_loss": valid_losses,
                            "epoch_train": train_steps,
                            "epoch_valid": valid_steps,
                        }
                        with open(
                            os.path.join(save_dir, "results.pkl"), "wb"
                        ) as rf:
                            pickle.dump(results, rf)
                        return

                except Exception as exc:
                    print(f"Eval failed at step {step}: {exc}")

            if step % save_every == 0:
                ckpt = os.path.join(
                    save_dir, f"lora_step_{step:07d}.safetensors"
                )
                save_lora_adapters(model, ckpt)
                save_lora_adapters(
                    model,
                    os.path.join(save_dir, "adapters.safetensors"),
                )

    # Save final adapter after reaching full iters
    final_ckpt = os.path.join(save_dir, "lora_final.safetensors")
    save_lora_adapters(model, final_ckpt)
    save_lora_adapters(model, os.path.join(save_dir, "adapters.safetensors"))

    results = {
        "train_loss": [float(v) for v in train_losses],
        "valid_loss": valid_losses,
        "epoch_train": train_steps,
        "epoch_valid": valid_steps,
    }
    with open(os.path.join(save_dir, "results.pkl"), "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LoRA training from a YAML config file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="conf/training.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        train_config = yaml.safe_load(f)

    train(**train_config)
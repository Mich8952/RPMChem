import itertools
import json
import os
import pickle
import shutil
from uuid import uuid4

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from transformers import AutoTokenizer 


"""
Inspiration/help for some code was taken from the mlx-lm package. This is an open source package that makes it easy to do LoRA fine tuning. However, we wanted to make it more customizable and permit more controllability so we coded it up in mlx. Note that mlx supposively is significantly faster than using pytorch with mps. This is also true because like pytorch MPS doesnt support 4 bit training but mlx permits it.
"""

from dataclasses_mlx import DataLoader, JSONLDataset
from models import causal_lm_loss, linear_to_lora_layers, load_pretrained_model, save_lora_adapters
from tokenizer_template import patch_chat_template_jinja, patch_tokenizer_config

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None


DEFAULT_SYSTEM_PROMPT = ""


def resolve_model_dir(model_dir_or_repo):
    if os.path.exists(model_dir_or_repo):
        return model_dir_or_repo
    # otherwise we load from hugging face!
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
    # look fro max position embedding first (hard limit) otherwise fall back and look for the model max length given in the tokenizer (would likely be the same, but I would prefer to use the model config instead)
    max_pos = model_config.get("max_position_embeddings")
    if isinstance(max_pos, int):
        return max_pos

    tok_max = getattr(tokenizer, "model_max_length")
    if isinstance(tok_max, int):
        return tok_max


"""
def catch_if_near_seq_len(max_seq_len, model_config, tokenizer):
    context_limit = get_context_limit(model_config, tokenizer)

    print(f"context_limit={context_limit} max_seq_len={max_seq_len}")
"""

def convert_batch_to_dct(batch):
    return {
        "input_ids": mx.array(batch["input_ids"]),
        "labels": mx.array(batch["labels"])}


def build_adapter_config( # most args are self explanatory. For example, eval_batches is how many batches we use during an eval call (so batch size * eval_batches many samples). This function converts the args into a dict form so that we can later seralize it into a json. 
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
    lora_alpha):

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


def evaluate(model, loader, max_batches): # load in the model, dataloader, and the max number of batches
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
    else:
        return sum(losses) / len(losses)


def copy_tokenizer_artifacts_from_orig_model(tokenizer_path, save_dir):
    if not os.path.exists(tokenizer_path):
        return

    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ):
        source_file = os.path.join(tokenizer_path, name)
        target_file = os.path.join(save_dir, name)
        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)


def bake_prompt_into_saved_chat_template(save_dir, system_prompt):
    """
    If we train with a prompt (we were experimenting before) then we need to put this new prompt into the tokenizer/chat template so that it loads it up during inference
    """
    tokenizer_config_path = os.path.join(save_dir, "tokenizer_config.json")
    chat_template_path = os.path.join(save_dir, "chat_template.jinja")
    updated_config = patch_tokenizer_config(tokenizer_config_path, system_prompt)
    updated_jinja = patch_chat_template_jinja(chat_template_path, system_prompt)
    if updated_config:
        print(f"updated prompt: {tokenizer_config_path}")
    else:
        print(f"unchanged: {tokenizer_config_path}. If we expected an update then please debug")
    if updated_jinja:
        print(f"updated prompt: {chat_template_path}")
    elif os.path.exists(chat_template_path):
        print(f"unchanged: {chat_template_path}. If we expected an update then please debug")


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
    seed):

    """
    Function to train the QLoRA adapters. Inputs should be self explanatory, but will document further in the future.
    I tried to make this as pytorch-thonic as possible
    """
    mx.random.seed(seed) # mlx has its own seed you gotta set

    model_dir = resolve_model_dir(model_dir)
    tokenizer_path = model_dir # load this from the model dir path (should come with the tokennizer per our original allow_pattersn setups)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token # depends on the model we use

    with open(f"{model_dir}/config.json", "r", encoding="utf-8") as f:
        model_config = json.load(f)

    if valid_jsonl == "split_from_train": # method so that it does a split from the train set instead at run time 

        train_ds = JSONLDataset(
        train_jsonl, 
        tokenizer,
        max_length=max_seq_len,
        apply_chat_template=apply_chat_template,
        mask_prompt=mask_prompt,
        system_prompt=system_prompt,
        split_prop=0.1765, # 17.65% of the original 85% (train) constitutes a final 70,15,15 split
        set_type = "train"
        )

        valid_ds = JSONLDataset(
            train_jsonl,
            tokenizer,
            max_length=max_seq_len,
            apply_chat_template=apply_chat_template,
            mask_prompt=mask_prompt,
            system_prompt=system_prompt,
            split_prop = 0.1765, # 17.65% of the original 85% (train) constitutes a final 70,15,15 split
            set_type = "valid"
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

    # I tried to make this as similar to pytorch as possible.

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
        shuffle=False, # TODO, looking back, we should maybe set this to true because the validation set is sampled each time we run an eval (so that evals do not take forever)
        seed=seed,
    )

    # now, load up the base model
    model, _ = load_pretrained_model(model_dir)
    model.freeze() # freeze the base parameters (we train LoRA parallel to these, but we dont want to update the base params)

    # loss tracking
    train_losses = []
    valid_losses = []
    valid_steps = []
    train_steps = []

    if num_layers > len(model.layers):
        raise ValueError(
            f"You asked for {num_layers} layers for LoRA, but the model only has {len(model.layers)} so you should reduce"
        )
    # convert linear layers to LoRA layers. Basically adding a parallel adapter.
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
        return causal_lm_loss(logits, cur_batch["labels"]) # compare logit prob distros to the true labels (one hot encoded).

    loss_and_grad = nn.value_and_grad(model, loss_fn) # this is like pytorch cost function but it also like automatically calls loss.backward() so it computes the gradients.

    # functionality for saving the prog
    os.makedirs(save_dir, exist_ok=True)
    adapter_config_path = f"{save_dir}/adapter_config.json"
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
            indent=4, # 4 spaces for each indent
        )
    print(f"saved={adapter_config_path}")

    # save the prompt if we use one (this is more for old tests I ran, but it could come in handy later, so i will keep it for now)
    prompt_config_path = f"{save_dir}/chem_prompt_config.json"
    with open(prompt_config_path, "w", encoding="utf-8") as f:
        json.dump(
            {"system_prompt": system_prompt},
            f,
            indent=2,
        )

    copy_tokenizer_artifacts_from_orig_model(tokenizer_path, save_dir)
    bake_prompt_into_saved_chat_template(save_dir, system_prompt)

    # also save logs (more useful before, but good to have nontheless)
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/run_{str(uuid4())}.log"
    log_file = open(log_path, "a", encoding="utf-8")

    train_iter = itertools.cycle(train_loader)

    for step in range(iters):
        batch = convert_batch_to_dct(next(train_iter))
        loss, grads = loss_and_grad(model, batch) # same as loss = cost(*) then loss.backward()

        optimizer.update(model, grads) # this is like pytorch optim.step(), but it just plans to update the params
        mx.eval(loss, model.parameters(), optimizer.state) # to actually update the params, you have to call mlx.eval as so.

        # some logging
        if step % 10 == 0 or step == 0:
            msg = f"step={step} train_loss={float(loss.item()):.6f}" # make it its own var because we want to print and log it
            
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

        if step % 30 == 0:
            train_steps.append(step)
            train_losses.append(loss.item())

        if step % eval_every == 0 and step != 0: 
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
                plt.savefig("train_curr_temp") # real time plotting
            except:
                print(f"stmh failed here")

        if step % save_every == 0:
            ckpt = f"{save_dir}/lora_step_{step:07d}.safetensors" # adding zeros so its a consistent form
            save_lora_adapters(model, ckpt)
            save_lora_adapters(model, f"{save_dir}/adapters.safetensors") # current adapter (if we want to run tests during training)
 
    # save final adapter when done (if we reach iters)
    final_ckpt = f"{save_dir}/lora_final.safetensors"
    save_lora_adapters(model, final_ckpt)
    save_lora_adapters(model, f"{save_dir}/adapters.safetensors")

    with open(f"{save_dir}/results.pkl", "wb") as f: # saving results that occured during training
        pickle.dump(
            {
                "train_loss": [float(v) for v in train_losses],
                "valid_loss": valid_losses,
                "epoch_train": train_steps,
                "epoch_valid": valid_steps,
            },
            f,
        )


    log_file.close()


if __name__ == "__main__":
    train(
        model_dir="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        train_jsonl="/Users/michaelmurray/Documents/GitHub/RPMChem/datasets/current_to_run_with_txt_name/train_noimpute_mega_joined_3txt_with_textbook_ids.jsonl",
        valid_jsonl="split_from_train",
        save_dir="adapters_manual",
        max_seq_len=5000,
        batch_size=1,
        iters=5000,
        eval_every=250,
        eval_batches=125,
        save_every=250,
        apply_chat_template=True,
        mask_prompt=True,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        lr=1e-5,
        weight_decay=0.0,
        lora_rank=16,
        lora_alpha=32.0,
        lora_dropout=0.0,
        num_layers=-1,
        seed=42,
    )




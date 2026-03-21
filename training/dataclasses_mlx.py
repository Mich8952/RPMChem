import json
import os
import random
from sklearn.model_selection import train_test_split

import numpy as np

# formats a record (prompt adn completion) as a simple instruction text block
# used when apply_chat_template=False
def format_example_plain(record, system_prompt=None):
    prompt = record.get("prompt", "")
    completion = record.get("completion", "")
    if system_prompt:
        return (
            f"### System:\n{system_prompt}\n\n"
            f"### Prompt:\n{prompt}\n\n### Completion:\n{completion}"
        )
    return f"### Prompt:\n{prompt}\n\n### Completion:\n{completion}"


# same as above but only formats up to the start of the completion and doesnt actualy put the completion
def format_prompt_only_plain(record, system_prompt=None):
    prompt = record.get("prompt", "")
    if system_prompt:
        return (
            f"### System:\n{system_prompt}\n\n"
            f"### Prompt:\n{prompt}\n\n### Completion:\n"
        )
    return f"### Prompt:\n{prompt}\n\n### Completion:\n"

# uses the tokenizer's apply_chat_template so the model sees its expected conversation format - this is what it was originally like trained on 
def format_example_chat(
    record,
    tokenizer,
    system_prompt=None,
):
    prompt = record.get("prompt", "")
    completion = record.get("completion", "")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
    )
    return tokenizer.apply_chat_template( # it has some weird "current date" thing but I dont think we update it because its supposed to always have it during inference. Its more of its own internal safegauard I believe.
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


# chat template version of format_prompt_only_plain. Again, no completion is present here
def format_prompt_only_chat(
    record,
    tokenizer,
    system_prompt=None,
):
    prompt = record.get("prompt", "")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True, # ensures its in the right format for assistant role.
    )


class JSONLDataset:
    def __init__(
        self,
        jsonl_path,
        tokenizer,
        max_length=1024,
        apply_chat_template=True,
        mask_prompt=True, # if True, only compute loss on the completion tokens (not the prompt)
        system_prompt=None,
        split_prop = None,
        set_type = None
    ):
        if split_prop is None and set_type is not None:
            raise ValueError("set_type should only be specified if split_prop is also specified")
        elif split_prop is not None and set_type is None:
            raise ValueError("set_type must be specified if split_prop is specified")
        self.path = os.path.abspath(f"{jsonl_path}")
        self.split_again = True if split_prop is not None else False
        self.split_prop = split_prop
        self.set_type = set_type
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.apply_chat_template = apply_chat_template
        self.mask_prompt = mask_prompt
        self.system_prompt = (system_prompt or "").strip() or None
        self.samples = self.load_items()
        

    def load_items(self):
        samples = []
        txt_ids = [] # if relevant
        truncated_count = 0
        use_chat_template = self.apply_chat_template and hasattr(self.tokenizer, "apply_chat_template")
        with open(self.path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if use_chat_template:
                    text = format_example_chat(record, self.tokenizer, self.system_prompt)
                    prompt_only = format_prompt_only_chat(record, self.tokenizer, self.system_prompt)
                else:
                    text = format_example_plain(record, self.system_prompt)
                    prompt_only = format_prompt_only_plain(record, self.system_prompt)
                token_ids_full = self.tokenizer.encode(text, add_special_tokens=True)
                prompt_ids = self.tokenizer.encode(prompt_only, add_special_tokens=True)
                was_truncated = len(token_ids_full) > self.max_length
                if was_truncated:
                    truncated_count += 1
                    if truncated_count <= 5: # arbitrary thresh, however, a truncation of a few tokens likely is not a huge deal, but in general we should increase the max_seq_len if this gets triggered
                        print(
                            f"WARNING Truncated sample at {self.path},{line_idx} "
                            f"(tokens={len(token_ids_full)} > max_length={self.max_length})."
                        )
                token_ids = token_ids_full[:self.max_length]
                loss_start = min(len(prompt_ids), len(token_ids))
                if len(token_ids) >= 2: # there previously was a couple corrupted samples of empty strings or single words, so this handles those (in reality I think these are fixed now in preprocessing)                    
                    samples.append((token_ids, loss_start))
                    txt_id = record.get("textbook_id")
                    if txt_id is not None:
                        txt_ids.append(txt_id)
        if truncated_count > 0:
            print(f"WARNING: {truncated_count} samples were truncated to max_length={self.max_length} ") # again we want to fix this immediately if this triggers

        if self.split_again:
            train_samples, test_samples = train_test_split(samples, test_size=self.split_prop, random_state=42, stratify=txt_ids)         
            if self.set_type == "train":
                samples = train_samples
            elif self.set_type == "valid":
                samples = test_samples
        return samples

    def __len__(self): #pytorch like __len__
        return len(self.samples)

    def __getitem__(self, idx): # can use the same like __get_item__ special method so that samples can bre indexed.
        return self.samples[idx]


class DataLoader: # pytorch like datalaoder
    def __init__(
        self,
        dataset,
        batch_size,
        pad_token_id,
        shuffle=True,
        drop_last=False,
        seed=42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.random_gen = random.Random(seed) 

    def __len__(self):
        n = len(self.dataset)
        return (n + self.batch_size-1) // self.batch_size # number of batches (pytorch len of dataloader gives this)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            self.random_gen.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            chunk = indices[start : start + self.batch_size]
            if self.drop_last and len(chunk) < self.batch_size:
                break
            batch = [self.dataset[i] for i in chunk]
            yield self.collate(batch)

    # pad all sequences in the batch to the same length, then build the labels array
    # padding positions in labels stay at -100 so the loss function ignores them
    def collate(self, batch):
        max_len = max(len(x[0]) for x in batch)

        input_ids = np.full((len(batch), max_len), self.pad_token_id, dtype=np.int32)
        labels = np.full((len(batch), max_len), -100, dtype=np.int32) 

        for i, (token_ids, loss_start) in enumerate(batch):
            n = len(token_ids)
            input_ids[i, :n] = token_ids
            if self.dataset.mask_prompt:
                labels[i, loss_start:n] = token_ids[loss_start:n] # only compute loss on completion tokens
            else:
                labels[i, :n] = token_ids # compute loss on everything (prompt and completion)

        return {"input_ids": input_ids,"labels": labels}

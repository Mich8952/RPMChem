import json
import os
import random

import numpy as np
from sklearn.model_selection import train_test_split


def format_example_plain(record, system_prompt=None):
    """Format a record as a plain instruction text block.

    Used when apply_chat_template=False.
    """
    prompt = record.get("prompt", "")
    completion = record.get("completion", "")
    if system_prompt:
        return (
            f"### System:\n{system_prompt}\n\n"
            f"### Prompt:\n{prompt}\n\n### Completion:\n{completion}"
        )
    return f"### Prompt:\n{prompt}\n\n### Completion:\n{completion}"


def format_prompt_only_plain(record, system_prompt=None):
    """Format only the prompt portion (no completion) as plain text."""
    prompt = record.get("prompt", "")
    if system_prompt:
        return (
            f"### System:\n{system_prompt}\n\n"
            f"### Prompt:\n{prompt}\n\n### Completion:\n"
        )
    return f"### Prompt:\n{prompt}\n\n### Completion:\n"


def format_example_chat(record, tokenizer, system_prompt=None):
    """Format a record using the tokenizer's apply_chat_template.

    Ensures the model sees the same conversation format it was
    originally trained on.
    """
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
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def format_prompt_only_chat(record, tokenizer, system_prompt=None):
    """Format only the prompt portion using the tokenizer's chat template."""
    prompt = record.get("prompt", "")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


class JSONLDataset:
    def __init__(
        self,
        jsonl_path,
        tokenizer,
        max_length=1024,
        apply_chat_template=True,
        mask_prompt=True,
        system_prompt=None,
        split_prop=None,
        set_type=None,
    ):
        """Dataset backed by a JSONL file.

        Args:
            jsonl_path: Path to the .jsonl file.
            tokenizer: HuggingFace tokenizer instance.
            max_length: Maximum token sequence length; longer samples
                are truncated.
            apply_chat_template: Whether to use the tokenizer's chat
                template format.
            mask_prompt: If True, only compute loss on completion
                tokens (not the prompt).
            system_prompt: Optional system prompt string.
            split_prop: Fraction of data to hold out as validation.
                Must be set together with set_type.
            set_type: One of "train" or "valid". Required when
                split_prop is provided.
        """
        if split_prop is None and set_type is not None:
            raise ValueError(
                "set_type should only be specified if split_prop is also specified"
            )
        if split_prop is not None and set_type is None:
            raise ValueError(
                "set_type must be specified if split_prop is specified"
            )

        self.path = os.path.abspath(jsonl_path)
        self.split_again = split_prop is not None
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
        txt_ids = []
        truncated_count = 0
        use_chat = (
            self.apply_chat_template
            and hasattr(self.tokenizer, "apply_chat_template")
        )

        with open(self.path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                if use_chat:
                    text = format_example_chat(
                        record, self.tokenizer, self.system_prompt
                    )
                    prompt_only = format_prompt_only_chat(
                        record, self.tokenizer, self.system_prompt
                    )
                else:
                    text = format_example_plain(record, self.system_prompt)
                    prompt_only = format_prompt_only_plain(
                        record, self.system_prompt
                    )

                token_ids_full = self.tokenizer.encode(
                    text, add_special_tokens=True
                )
                prompt_ids = self.tokenizer.encode(
                    prompt_only, add_special_tokens=True
                )

                was_truncated = len(token_ids_full) > self.max_length
                if was_truncated:
                    truncated_count += 1
                    if truncated_count <= 5:
                        print(
                            f"WARNING Truncated sample at "
                            f"{self.path},{line_idx} "
                            f"(tokens={len(token_ids_full)} > "
                            f"max_length={self.max_length})."
                        )

                token_ids = token_ids_full[: self.max_length]
                loss_start = min(len(prompt_ids), len(token_ids))

                # Filter out corrupted/empty samples (< 2 tokens)
                if len(token_ids) >= 2:
                    samples.append((token_ids, loss_start))
                    txt_id = record.get("textbook_id")
                    if txt_id is not None:
                        txt_ids.append(txt_id)

        if truncated_count > 0:
            print(
                f"WARNING: {truncated_count} samples were truncated to "
                f"max_length={self.max_length}"
            )

        if self.split_again:
            train_samples, test_samples = train_test_split(
                samples,
                test_size=self.split_prop,
                random_state=42,
                stratify=txt_ids,
            )
            if self.set_type == "train":
                samples = train_samples
            elif self.set_type == "valid":
                samples = test_samples

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class DataLoader:
    """PyTorch-style DataLoader for JSONLDataset."""

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
        return (n + self.batch_size - 1) // self.batch_size

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

    def collate(self, batch):
        """Pad sequences and build label arrays.

        Padding positions in labels are set to -100 so the loss
        function ignores them.
        """
        max_len = max(len(x[0]) for x in batch)

        input_ids = np.full(
            (len(batch), max_len), self.pad_token_id, dtype=np.int32
        )
        labels = np.full((len(batch), max_len), -100, dtype=np.int32)

        for i, (token_ids, loss_start) in enumerate(batch):
            n = len(token_ids)
            input_ids[i, :n] = token_ids
            if self.dataset.mask_prompt:
                # Only compute loss on completion tokens
                labels[i, loss_start:n] = token_ids[loss_start:n]
            else:
                # Compute loss on prompt + completion
                labels[i, :n] = token_ids

        return {"input_ids": input_ids, "labels": labels}
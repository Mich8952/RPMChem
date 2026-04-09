# preprocessing/add_reasoning_context.py

"""
Generates reasoning traces to augment question-solution training examples.

Ideally, one would use a powerful LLM, however this project uses a local
gpt-oss-120b model via LM Studio to avoid API costs.
"""

import json
import os

import lmstudio as lms

from prompts import REASONING_SYSTEM_PROMPT, REASONING_USER_TEMPLATE


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "gpt-oss-120b"
TEMPERATURE = 0.0
MAX_TOKENS = 3000


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class SplitProcessor:
    """Loads a JSONL split, augments each row with model-generated reasoning,
    and writes the result to a new JSONL file."""

    def __init__(self):
        self.model = lms.llm(MODEL)

    def process_split(self, input_path: str, output_path: str) -> None:
        rows = self.load_jsonl(input_path)
        out_rows = []
        failures = 0

        for idx, row in enumerate(rows, start=1):
            try:
                reasoning = self.send(
                    prompt=row["prompt"], completion=row["completion"]
                )
                augmented = self.compose_augmented_completion(
                    reasoning, row["completion"]
                )
                out_rows.append({"prompt": row["prompt"], "completion": augmented})
            except Exception as e:
                failures += 1
                print(f"Row {idx} generation failed: {e}")

            if idx % 10 == 0:
                print(f"{idx}/{len(rows)} done ({idx / len(rows) * 100:.1f}%)")

        self.write_jsonl(output_path, out_rows)
        print(f"Wrote {len(out_rows)} rows to {output_path} (failures={failures})")

    def load_jsonl(self, path: str) -> list[dict]:
        """Read a JSONL file and return a list of prompt/completion dicts."""
        rows = []
        path_str = os.fspath(path)
        with open(path_str, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj.get("prompt", "")
                completion = obj.get("completion", "")
                if not isinstance(prompt, str) or not isinstance(completion, str):
                    raise ValueError(
                        f"Invalid prompt/completion type at {path_str}:{line_idx}"
                    )
                rows.append({"prompt": prompt, "completion": completion})
        return rows

    def build_user_prompt(self, prompt: str, solution: str) -> str:
        """Format the user-facing prompt using the shared template."""
        return REASONING_USER_TEMPLATE.format(prompt=prompt, completion=solution)

    def send(self, prompt: str, completion: str) -> str:
        """Send a prompt+solution pair to the model and return reasoning text."""
        chat = lms.Chat(REASONING_SYSTEM_PROMPT)
        chat.add_user_message(self.build_user_prompt(prompt, completion))
        prediction = self.model.respond(
            chat,
            config={"temperature": TEMPERATURE, "maxTokens": MAX_TOKENS},
        )
        return prediction.content or ""

    def compose_augmented_completion(self, reasoning: str, solution: str) -> str:
        """
        Combine reasoning and solution into a single completion string.

        The model is trained to produce reasoning followed by the final
        solution, so both are concatenated here under labelled headers.
        """
        if not reasoning:
            raise ValueError(
                "Empty reasoning returned by model — skipping this sample."
            )
        return f"Reasoning:\n{reasoning}\n\nSolution:\n{solution}"

    def write_jsonl(self, path: str, rows: list[dict]) -> None:
        """Write a list of dicts to a JSONL file, creating parent dirs if needed."""
        path_str = os.fspath(path)
        parent_dir = os.path.dirname(path_str)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(path_str, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    input_train = "datasets/current_to_run/train_noimpute.jsonl"
    input_valid = "datasets/current_to_run/valid_noimpute.jsonl"
    output_train = "datasets/current_to_run/train_reasoning_n.jsonl"
    output_valid = "datasets/current_to_run/valid_reasoning_n.jsonl"

    sp = SplitProcessor()
    sp.process_split(input_train, output_train)
    sp.process_split(input_valid, output_valid)


# ---------------------------------------------------------------------------
# End of file!
# ---------------------------------------------------------------------------
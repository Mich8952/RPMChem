"""
Main purpose of this file is to add reasoning to bridge questions and their solutions. The idea is that hopefully the LLM used here will be faithful and produce strong reasoning that would help another model learn how to reason from question to solution.

In this case we want to use a smart LLM (but I also don't want to spend money on API calls), so I will use a local gpt-oss-120b model
"""

import json
from pathlib import Path

import lmstudio as lms

SYSTEM_PROMPT = (
    "You are an expert chemistry tutor. Given a chemistry question and a ground-truth final "
    "solution, generate only the reasoning steps that would lead to that solution. "
    "Be numerically and chemically consistent. Do not restate the final solution verbatim."
)

MODEL = "gpt-oss-120b"
API_KEY = "lm-studio"
TEMPERATURE = 0.2
MAX_TOKENS = 1000 # will fix this if we run into issues with cutoff

#model = lms.llm(MODEL)



class SplitProcessor:
    def __init__(self):
        # only making this class based so that we can load the model only when this is called
        self.model = lms.llm(MODEL)
            
    def process_split(self, input_path, output_path):
        rows = self.load_jsonl(input_path)
        out_rows = []
        failures = 0
        for idx, row in enumerate(rows, start=1):
            try:
                reasoning = self.send(prompt=row["prompt"], completion=row["completion"])
                
                augmented = self.compose_augmented_completion(reasoning, row["completion"])
                out_rows.append({"prompt": row["prompt"], "completion": augmented})


            except Exception as e:
                failures += 1
                print(f"{idx} generation failed: {e}")

            print(idx/len(rows)*100)
            if idx % 10 == 0:
                print(f"{idx}/{len(rows)} done")

        self.write_jsonl(output_path, out_rows)
        print(f"Wrote {len(out_rows)} rows to {output_path} (failures={failures})")

    def load_jsonl(self, path):
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj.get("prompt", "")
                completion = obj.get("completion", "")
                if not isinstance(prompt, str) or not isinstance(completion, str): # have to handle cases where it for some reason didnt save the formatting correctly before (likely an issue with my get_jsons script. Will debug in future)
                    raise ValueError(f"Invalid prompt/completion at {path}:{line_idx}")
                rows.append({"prompt": prompt, "completion": completion})

        return rows


    def build_user_prompt(self, prompt, solution):

        to_writeout = (
            "Question:\n"
            f"{prompt}\n\n"
            "Ground-truth final solution:\n"
            f"{solution}\n\n"
            "Task:\n"
            "Write only the reasoning steps that lead to the provided final solution.\n"
            "Requirements:\n"
            "1) Keep equations and unit logic explicit.\n"
            "2) Keep it concise but complete enough to reproduce the solution.\n"
            "3) Do NOT include a final-answer section.\n"
            "4) Do NOT add JSON or markdown fences.\n"
        )

        return to_writeout


    def send(self, prompt, completion):
        chat = lms.Chat(SYSTEM_PROMPT)
        chat.add_user_message(self.build_user_prompt(prompt, completion))

        prediction = self.model.respond(
            chat,
            config={"temperature": TEMPERATURE,"maxTokens": MAX_TOKENS},
        )

        text = prediction.content
        return text or "" # if text is none then we just send the empty string (handle later)


    def compose_augmented_completion(self, reasoning, solution):
        """
        Augmented message simply means that we replace the old "completion" with reasoning and completion (all in one text block). This is because we want to train the model to learn the reasoning AND solution.
        """
        if reasoning is None or reasoning == "":
            raise ValueError("Model failed to make reasoning for some reason, ignore this entire sample")
        reasoning_text = reasoning
        solution_text = solution
        return f"Reasoning:\n{reasoning_text}\n\nSolution:\n{solution_text}"


    def write_jsonl(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n") # new the newline because its a jsonl.\


if __name__ == "__main__":
    input_train = "datasets/current_to_run/train_noimpute.jsonl"
    input_valid = "datasets/current_to_run/valid_noimpute.jsonl"
    output_train = "datasets/current_to_run/train_reasoning_n.jsonl"
    output_valid = "datasets/current_to_run/valid_reasoning_n.jsonl"

    train_in = Path(input_train)
    valid_in = Path(input_valid)
    train_out = Path(output_train)
    valid_out = Path(output_valid)

    sp = SplitProcessor()
    sp.process_split(train_in,train_out)
    sp.process_split(valid_in,valid_out)

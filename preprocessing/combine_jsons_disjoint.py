# preprocessing/combine_jsons_disjoint.py

"""
Builds a joint JSONL dataset from separate question and solution JSON files.
"""


import json

import pandas as pd


# ---------------------------------------------------------------------------
# Combiner
# ---------------------------------------------------------------------------

class Combiner:
    """Merges extracted question and solution JSON files into a single JSONL."""

    def __init__(
        self,
        question_json_path: str,
        solution_json_path: str,
        output_json_path: str,
    ):
        self.question_json_path = question_json_path
        self.solution_json_path = solution_json_path
        self.output_json_path = output_json_path

    @staticmethod
    def load_and_convert_to_df(json_path: str) -> pd.DataFrame:
        """
        Load a records JSON file produced by extract_items_from_pdf and
        return a DataFrame with columns [question_num, question_text] or
        [question_num, answer_text] depending on the record schema.

        Raises:
            ValueError: If the records list is empty.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = data.get("records", [])
        if not records:
            raise ValueError(f"No records found in {json_path}")

        is_questions = "question_text" in records[0]
        text_key = "question_text" if is_questions else "answer_text"

        return pd.DataFrame(
            {
                "question_num": [r["question_num"] for r in records],
                text_key: [r[text_key] for r in records],
            }
        )

    def __call__(self) -> str:
        """
        Merge question and solution DataFrames on question_num, write the
        result to a JSONL file, and return the output path.
        """
        question_df = self.load_and_convert_to_df(self.question_json_path)
        solution_df = self.load_and_convert_to_df(self.solution_json_path)

        combined_df = pd.merge(question_df, solution_df, on="question_num")
        combined_df["valid"] = True
        combined_df.columns = ["question_num", "prompt", "completion", "valid"]
        combined_df = combined_df[["valid", "question_num", "prompt", "completion"]]

        with open(self.output_json_path, "w", encoding="utf-8") as f:
            combined_df.to_json(f, orient="records", lines=True)

        return self.output_json_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    question_json_path = "datasets/processed_real/pdf1_json.json"
    solution_json_path = "datasets/processed_real/pdf2_json.json"
    output_json_path = "datasets/processed_real/joined_disjoint_textbook.jsonl"

    combiner = Combiner(question_json_path, solution_json_path, output_json_path)
    combiner()


# ---------------------------------------------------------------------------
# End of file!
# ---------------------------------------------------------------------------
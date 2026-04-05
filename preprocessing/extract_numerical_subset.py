# preprocessing/extract_numerical_subset.py

"""
Extracts single final numerical values and units from processed samples.
"""

import ast
import lmstudio as lms
import json
import re
from tqdm import tqdm
import pandas as pd

class NumberExtractor:
    """
    Class that tries to extract numbers from questions. Uses a open souce LLM to do so because I don't want to pay of API calls (might affect quality, but using a relatively big model to compensate)

    Args:
        file_dir: str that specifies the dir to the jsonl you want to process
        model_dir: str that specifies the LMStudio model if desired
    """
    DEFAULT_EXTRACTION_PROMPT = """You are given a question and its corresponding answer.
Your task is to extract a SINGLE final numerical value and its unit (including SI prefix) if one exists.

Rules:
1. If the answer contains exactly one unambiguous final numerical value, output it.
2. Preserve unit text and SI prefix exactly when present (examples: MW, kJ, cm^2, m s^-1, %).
3. If the result is dimensionless, set unit to NA.
4. If there is no numerical value, more than one possible final value, a range, or ambiguity, set both value and unit to NA.
5. If the question/answer pair is not relevant to a single final answer (i.e., chemistry formula balancing), set both to NA.
6. Convert symbolic final forms to decimal when feasible.
7. Output STRICT JSON only with this shape:
{"value": "<number or NA>", "unit": "<unit or NA>"}

Do not include any explanation or extra text."""

    def __init__(
    self,
    file_dir: str | None = None,
    model_dir: str = "openai/gpt-oss-120b",
    output_csv: str = "datasets/numerical_prompts_real/validation_units_new.csv",
    ):
        self.model = lms.llm(model_dir)
        self.all_preds = []
        self.all_samples = []
        self.all_completions = []
        self.all_values = []
        self.all_units = []
        self.file_dir = file_dir
        self.output_csv = output_csv

    @staticmethod
    def convert_to_float(output):
        try:
            str_text = str(output).strip().replace(",", "")
            if str_text.upper() == "NA" or str_text == "":
                return None
            else:
                return float(str_text)
        except Exception:
            return None

    @staticmethod
    def convert_to_unit(output):
        try:
            str_text = str(output).strip()
            if str_text.upper() == "NA" or str_text == "":
                return None
            else:
                return str_text
        except Exception:
            return None


    def form_pred(self, sample, prompt=None):
        """
        Method that processes a given sample (you can put a custom prompt if needed  
        """
        if prompt is None:
            prompt = self.DEFAULT_EXTRACTION_PROMPT
        chat = lms.Chat(prompt)

        chat.add_user_message(str(sample))

        prediction = self.model.respond(chat, config={'temperature': 0})

        parsed_pred = prediction.content.split("final<|message|>")[-1].strip() # skip the reasoning part, only care about the final answer that it gives

        ## Extract the unit now
        text = str(parsed_pred).strip()
        obj = None
        try:
            obj = json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                block = match.group(0)
                try:
                    obj = json.loads(block) 
                except Exception:
                    try:
                        obj = ast.literal_eval(block) # if it returns a dict instead of json
                    except Exception:
                        obj = None

        if not isinstance(obj, dict):
            value_match = re.search(r"value\s*[:=]\s*([^\n]+)", text, flags=re.IGNORECASE)
            unit_match = re.search(r"unit\s*[:=]\s*([^\n]+)", text, flags=re.IGNORECASE)
            value = self.convert_to_float(value_match.group(1).strip() if value_match else None)
            unit = self.convert_to_unit(unit_match.group(1).strip() if unit_match else None)
            return value, unit

        value = self.convert_to_float(obj.get("value"))
        unit = self.convert_to_unit(obj.get("unit"))


        return value, unit

    def run_all(self):
        skip_counter = 0
        if self.file_dir is None:
            self.file_dir = input("Please enter the directory to the file that you would like to process: ")
        with open(self.file_dir, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            try:
                sample = json.loads(line)
                if "prompt" in sample and "completion" in sample:
                    value, unit = self.form_pred(sample)
                    if value is None:
                        raise Exception("no number")
                    self.all_samples.append(sample["prompt"])
                    self.all_values.append(value)
                    self.all_units.append(unit)
            except Exception:
                skip_counter+=1
                print(skip_counter)
                
        

        df = pd.DataFrame()
        df["prompt"] = self.all_samples
        df["all_pred"] = self.all_values
        df["all_pred_value"] = self.all_values
        df["all_pred_unit"] = [u if u is not None else "NA" for u in self.all_units]
        df.to_csv(self.output_csv)
                
if __name__ == "__main__":
    NumberExtractor(
        "datasets/current_to_run/valid_noimpute.jsonl"
    ).run_all()


# ---------------------------------------------------------------------------
# End of file!
# ---------------------------------------------------------------------------
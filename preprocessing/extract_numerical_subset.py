import lmstudio as lms
from openai import OpenAI
import os
import json
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
            Your task is to extract a SINGLE final numerical value from the answer IF one exists.

            Rules:
            1. If the answer contains exactly one unambiguous final numerical value, return only that number.
            2. Ignore units, formatting, commas, and explanatory text.
            3. If there is no numerical value, more than one possible final value, a range, or any ambiguity, return NA.
            4. If the question/answer pair is not relevant to a single final answer (i.e., chemistry formula balancing) then return NA
            5. Your output must be either:
                - The number (digits and decimal point only), or
                - NA
                

            Do not include any explanation, reasoning, commentary, or additional text.
            Output strictly the number or NA.
            Be very decisive; never put 'some number or NA'"""

    def __init__(self, file_dir = None, model_dir = "openai/gpt-oss-20b"):
        self.model = lms.llm(model_dir)
        self.all_preds = []
        self.all_samples = []
        self.all_completions = []
        self.file_dir = file_dir

    def form_pred(self, sample, prompt=None):
        """
        Method that processes a given sample (you can put a custom prompt if needed  
        """
        chat = lms.Chat(prompt or self.DEFAULT_EXTRACTION_PROMPT)

        chat.add_user_message(str(sample))

        prediction = self.model.respond(chat)

        #self.all_preds.append(prediction.content)
        parsed_pred = prediction.content.split("final<|message|>")[-1].strip() # skip the reasoning part, only care about the final answer that it gives
        self.all_preds.append(parsed_pred)

        return parsed_pred

    def run_all(self):
        if self.file_dir is None:
            self.file_dir = input("Please enter the directory to the file that you would like to process: ")
        with open(self.file_dir, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            try:
                sample = json.loads(line)
                if "prompt" in sample and "completion" in sample:
                    self.form_pred(sample)
                    self.all_samples.append(sample['prompt']) # want to also store the original prompt and completion
                    self.all_completions.append(sample['completion'])
            except json.JSONDecodeError:
                continue
        

        df = pd.DataFrame()
        df['prompt'] = self.all_samples
        df['completion'] = self.all_completions
        df['all_pred'] = self.all_preds

        df.to_csv("datasets/numerical_prompts_real/validation.csv")


if __name__ == "__main__":
    NumberExtractor("/Users/michaelmurray/Documents/GitHub/RPMChem/datasets/current_to_run/valid.jsonl").run_all()
                

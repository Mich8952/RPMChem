import pandas as pd
from tqdm import tqdm
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import sys

sys.path.append("/Users/michaelmurray/Documents/GitHub/RPMChem/preprocessing")
from extract_numerical_subset import NumberExtractor


"""
Need to fix
- No handling of units currently (maybe get the LLM to do it, but I don't like involving other LLMs so much)

For example
i=11 has complex units

On average the results will be okay to interpret for the interm report, but we should improve this in future.
"""

mx.random.seed(42)
 
ne = NumberExtractor()
NUMERICAL_EXTRACTION_PROMPT = """You are given a question and its corresponding answer. 
Your task is to extract a SINGLE final numerical value from the answer IF one   exists.

Rules:
1. If the answer contains exactly one unambiguous final numerical value, return only that number.
2. Ignore units, formatting, commas, and explanatory text.
3. If multiple numbers appear, choose the one that best matches the main requested final answer.
4. Prefer a value explicitly presented as the final result (for example after phrases like final answer, therefore, hence, or at the end of a calculation).
5. If there are multiple intermediate values, choose the most salient concluding value rather than NA.
6. If there is no numerical value, only ranges/intervals, or no defensible final numeric answer, return NA.
7. If the question asks for several independent final values and no single main value is identifiable, return NA.
8. If the question/answer pair is not relevant to a single final answer (i.e., chemistry formula balancing) then return NA.
9. Your output must be either:
    - The number (digits and decimal point only), or
    - NA
10. Do not include any explanation, reasoning, commentary, or additional text.
11. Output strictly the number or NA.
12. Be decisive and do not default to NA only because extra numbers are present.
13. You should convert symbolic answers (i.e., 1/√(2π) = 0.3989422804) to a number if possible"""

def extract_final_ans(question, answer): # grab final answer
    extraction_input = (
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}"
    )
    pred = ne.form_pred(extraction_input, prompt=NUMERICAL_EXTRACTION_PROMPT)
    if pred is None:
        return None

    pred = str(pred).strip()
    if pred.upper() == "NA" or pred == "":
        return None
    return float(pred)


class ModelComparatorNumerical: # class to compare models (this is a misnomer now because we are just extracting numbers for now)
    def __init__(self, dir_to_numerical_samples):
        df = pd.read_csv(dir_to_numerical_samples)
        self.samples_without = df[(~df['all_pred'].isna()) & (df['all_pred'] != 0)].reset_index(drop=True)
        self.all_prompts = []
        self.ground_truths = []

        self.model1_ans = []
        self.model2_ans = []

    def compare(self, model_dir1, model_dir2):
        model_1, tokenizer1 = load(model_dir1)
        model_2, tokenizer2 = load(model_dir2)


        for i in tqdm(range(len(self.samples_without))):

            word_prompt = self.samples_without.iloc[i]['prompt']

        
            ground_truth = float(self.samples_without.iloc[i]['all_pred'])

            messages1 = [{"role": "user", "content": word_prompt}]
            messages2 = [
                {"role": "system", "content": "Reasoning:\n"}, # to fit the training template
                {"role": "user", "content": word_prompt},
            ]
            prompt1 = tokenizer1.apply_chat_template(
                messages1, add_generation_prompt=True
            )
            prompt2 = tokenizer2.apply_chat_template(
                messages2, add_generation_prompt=True
            )

            text1 = generate(
                model_1,
                tokenizer1,
                prompt=prompt1,
                verbose=False,
                max_tokens=5000,
                sampler=make_sampler(temp=0.4),
            )
            text2 = generate(
                model_2,
                tokenizer2,
                prompt=prompt2,
                verbose=False,
                max_tokens=5000,
                sampler=make_sampler(temp=0.4),
            )

            try:
                try:
                    text2 = text2.split("Solution:\n", 1)[1] # force model to use soln if it doesnt abide by the format (this never happens anymore)
                except Exception:
                    if isinstance(prompt2, str):
                        prompt2_text = prompt2
                    else:
                        prompt2_text = tokenizer2.decode(prompt2)
                    recovery_prompt2 = prompt2_text + text2.rstrip() + "\n\nSolution:\n"
                    recovery_completion2 = generate(
                        model_2,
                        tokenizer2,
                        prompt=recovery_prompt2,
                        verbose=False,
                        max_tokens=1000,
                        sampler=make_sampler(temp=0.4),
                    )
                    recovered_text2 = text2.rstrip() + "\n\nSolution:\n" + recovery_completion2.lstrip() # grab the solution part (dont care about reasoning currently)
                    text2 = recovered_text2.split("Solution:\n", 1)[1]

                text1_ans = extract_final_ans(word_prompt, text1)
                text2_ans = extract_final_ans(word_prompt, text2)
                

                self.all_prompts.append(word_prompt)
                self.ground_truths.append(ground_truth)
                self.model1_ans.append(text1_ans)
                self.model2_ans.append(text2_ans)
                

                # WARNING doesnt consider units tho
                # might have to use a smarter LLM

                print(len(self.model1_ans), len(self.model2_ans))


            except Exception as e:
                print(f"fail: {e}")

        print(len(self.model1_ans), len(self.model2_ans))
    
    def save_results(self):
        if len(self.model1_ans) == 0 or len(self.model2_ans) == 0:
            print("No results to save, please run the compare method first") 
            return
        df = pd.DataFrame()
        df['prompt'] = self.all_prompts
        df['ground_truth'] = self.ground_truths
        df['model1_ans'] = self.model1_ans
        df['model2_ans'] = self.model2_ans

        df.to_csv("analysis/results/numerical_comparison_with_prompt.csv") # save to csv for stat processing later
    
if __name__ == "__main__":
    MCN = ModelComparatorNumerical("/Users/michaelmurray/Documents/GitHub/RPMChem/datasets/numerical_prompts_real/validation.csv")
    m1 = "/Users/michaelmurray/.lmstudio/models/personal/8b_noLora"
    m2 = "/Users/michaelmurray/.lmstudio/models/personal/fuse_model_8b_qlora_manual"
    MCN.compare(m1,m2)
    MCN.save_results()



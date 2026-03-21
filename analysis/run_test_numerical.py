import pandas as pd
from tqdm import tqdm
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import sys

sys.path.append("/Users/michaelmurray/Documents/GitHub/RPMChem/preprocessing")

from extract_numerical_subset import NumberExtractor

mx.random.seed(42)
 
ne = NumberExtractor()
NUMERICAL_EXTRACTION_PROMPT = """You are given a question and its corresponding answer.
Your task is to extract a SINGLE final numerical value and its unit (including SI prefix) if one exists.

Rules:
1. If the answer contains exactly one unambiguous final numerical value, output it.
2. Preserve unit text and SI prefix exactly when present (examples: MW, kJ, cm^2, m s^-1, %, mol L^-1).
3. If multiple numbers appear, choose the one that best matches the main requested final answer.
4. Prefer a value explicitly presented as the final result.
5. If there is no numerical value, only ranges/intervals, or no defensible final numeric answer, set value and unit to NA.
6. If the question asks for several independent final values and no single main value is identifiable, set value and unit to NA.
7. If the question/answer pair is not relevant to a single final answer (i.e., chemistry formula balancing), set value and unit to NA.
8. Convert symbolic final forms to decimal when feasible.
9. If the result is dimensionless, set unit to NA.
10. Output STRICT JSON only with this shape:
{"value": "<number or NA>", "unit": "<unit or NA>"}

Do not include any explanation or extra text."""



UNIT_CONVERSION_PROMPT = """You are converting one numerical value between units.

Input fields are:
- value
- from_unit
- to_unit

Rules:
1. Return NA if conversion is impossible, ambiguous, or dimensionally inconsistent.
2. Handle SI prefixes and common chemistry/science units.
3. Treat 'dimensionless' as a valid unit state.
4. Output STRICT JSON only with this shape:
{"value": "<number or NA>"}

Do not include any explanation or extra text."""

def extract_final_ans(question, answer): # grab final answer
    extraction_input = f"Question:\n{question}\n\nAnswer:\n{answer}"
    return ne.form_pred(extraction_input, prompt=NUMERICAL_EXTRACTION_PROMPT)




def normalize_unit_text(unit):
    if unit is None:
        return None
    s = str(unit).strip()
    if s == "" or s.upper() == "NA" or s.lower() == "nan":
        return None
    return s

def convert_to_target_unit(value,curr_unit,target_unit):
    if value is None:
        return None
    
    curr_unit_normed = normalize_unit_text(curr_unit)
    target_unit_normed = normalize_unit_text(target_unit)

    if curr_unit_normed is None or target_unit_normed is None:
        return value

    if curr_unit_normed == target_unit_normed:
        return value # already at the correct unit
    

    conversion_input = ( # input to the LLM doing. converison
        f"value: {value}\n"
        f"from_unit: {curr_unit_normed}\n"
        f"to_unit: {target_unit_normed}"
    )

    converted_value, _ = ne.form_pred(conversion_input, prompt=UNIT_CONVERSION_PROMPT)
    return converted_value




class ModelComparatorNumerical: # class to compare models (this is a misnomer now because we are just extracting numbers for now)
    def __init__(self, dir_to_numerical_samples):
        df = pd.read_csv(dir_to_numerical_samples)
        self.samples_without = df[(~df['all_pred'].isna()) & (df['all_pred'] != 0)].reset_index(drop=True)

        self.all_prompts = []
        self.ground_truths = []
        self.ground_truth_units = []

        self.model1_ans = []
        self.model2_ans = []

        self.model1_values_raw = []
        self.model1_units_raw = []
        self.model1_values_converted = []

        self.model2_values_raw = []
        self.model2_units_raw = []
        self.model2_values_converted = []

    def compare(self, model_dir1, model_dir2):
        model_1, tokenizer1 = load(model_dir1)
        model_2, tokenizer2 = load(model_dir2)


        for i in tqdm(range(len(self.samples_without))):
            try:
                curr_row = self.samples_without.iloc[i]
                word_prompt = curr_row['prompt']

                ground_truth = float(curr_row['all_pred'])
                ground_truth_unit = normalize_unit_text(curr_row["all_pred_unit"])

                messages1 = [{"role": "user", "content": word_prompt}]
                messages2 = [
                    {"role": "user", "content": word_prompt},
                ]

                prompt1 = tokenizer1.apply_chat_template(
                    messages1, add_generation_prompt=True
                )
                prompt2 = tokenizer2.apply_chat_template(
                    messages2, add_generation_prompt=True
                )

                try:
                    text1 = generate(
                        model_1,
                        tokenizer1,
                        prompt=prompt1,
                        verbose=False,
                        max_tokens=5000,
                        sampler=make_sampler(temp=1),
                    )
                except:
                    text1 = ""

                try:
                    text2 = generate(
                        model_2,
                        tokenizer2,
                        prompt=prompt2,
                        verbose=False,
                        max_tokens=5000,
                        sampler=make_sampler(temp=1),
                    )
                except:
                    text2 = ""

            
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

                try:
                    text1_value_raw, text1_unit_raw = extract_final_ans(word_prompt, text1)
                except:
                    text1_value_raw, text1_unit_raw = None, None
                try:
                    text2_value_raw, text2_unit_raw = extract_final_ans(word_prompt, text2)
                except:
                    text2_value_raw, text2_unit_raw = None, None

                text1_converted = convert_to_target_unit(text1_value_raw, text1_unit_raw, ground_truth_unit)
                text2_converted = convert_to_target_unit(text2_value_raw, text2_unit_raw, ground_truth_unit)

                self.all_prompts.append(word_prompt)
                self.ground_truths.append(ground_truth)
                self.ground_truth_units.append(ground_truth_unit if ground_truth_unit is not None else "NA")

                self.model1_values_raw.append(text1_value_raw)
                self.model1_units_raw.append(text1_unit_raw if text1_unit_raw is not None else "NA")
                self.model1_values_converted.append(text1_converted)
                self.model1_ans.append(text1_converted)

                self.model2_values_raw.append(text2_value_raw)
                self.model2_units_raw.append(text2_unit_raw if text2_unit_raw is not None else "NA")
                self.model2_values_converted.append(text2_converted)
                self.model2_ans.append(text2_converted)

                print('\n')
                print(self.model1_ans[-1], self.model2_ans[-1], self.ground_truths[-1])
                print('\n')

                print(len(self.model1_ans), len(self.model2_ans))
                if self.model1_ans is not None:
                    print("")

            except Exception as e:
                print(f"fail: {e}")

        print(len(self.model1_ans), len(self.model2_ans))
    
    def save_results(self):
        if len(self.model1_ans) == 0 or len(self.model2_ans) == 0:
            print("No results to save, please run the compare method first")
            return

        df = pd.DataFrame()
        df["prompt"] = self.all_prompts
        df["ground_truth"] = self.ground_truths
        df["ground_truth_unit"] = self.ground_truth_units

        df["model1_value_raw"] = self.model1_values_raw
        df["model1_unit_raw"] = self.model1_units_raw
        df["model1_converted_value"] = self.model1_values_converted
        df["model1_ans"] = self.model1_ans

        df["model2_value_raw"] = self.model2_values_raw
        df["model2_unit_raw"] = self.model2_units_raw
        df["model2_converted_value"] = self.model2_values_converted
        df["model2_ans"] = self.model2_ans

        #df.to_csv("analysis/results/numerical_comparison_3txt_noignoreunits_train2txt_apply3txt.csv")
        df.to_csv("analysis/results/temp.csv")
    
if __name__ == "__main__":
    MCN = ModelComparatorNumerical("/Users/michaelmurray/Documents/GitHub/RPMChem/datasets/numerical_prompts_real/validation.csv")
    m1 = "/Users/michaelmurray/.lmstudio/models/personal/8b_noLora"
    m2 = "/Users/michaelmurray/.lmstudio/models/personal/fuse_model_8b_qlora_manual_NEW"
    MCN.compare(m1, m2)
    MCN.save_results()


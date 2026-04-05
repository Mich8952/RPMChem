## Compare vanilla LLM to the fine-tuned LLM

import warnings
import numpy as np
import pandas as pd
import mlx.core as mx
import copy
from rouge_score import rouge_scorer
import re

warnings.filterwarnings("ignore", message=".*UNEXPECTED.*") #annoying warnings
warnings.filterwarnings("ignore", message=".*interactive.*")
warnings.filterwarnings("ignore", category=UserWarning)
from transformers import logging
logging.set_verbosity_error()
logging.disable_progress_bar()
logging.set_verbosity(logging.CRITICAL)

from datetime import datetime

from mlx_lm import load, generate
from tqdm import tqdm
from bert_score import score
import json

mx.random.seed(42)

sys_prompt_model1 = """
You are a textbook solution solver.
Use this exact output format: "Reasoning:\n<your reasoning>\n\nSolution:\n<your final answer>".
First provide reasoning, then provide solution in textbook solution-manual style.
"""


class ModelComparatorSemantics:
    def __init__(self, dataset_dir = "datasets/processed/valid.jsonl", model_type = "allenai/scibert_scivocab_uncased"):
        with open(dataset_dir, "r") as f:
            self.dataset = [json.loads(line) for line in f]
        # new shuffle so that we shuffle the dataset (i.e., if there is inherent order to the textbooks)
        # in reality, this does not matter at all because we run through the entire dataset and compute the average
        # however, it means that the reported metrics would be a proper estimate of the full training if we are like halfway through
        idx = np.random.permutation(len(self.dataset))
        self.dataset = np.array(self.dataset)[idx]
        
        self.model_type = model_type
        self.all_prompts = []
        self.all_ground_truth_completions = []
        self.all_model1_completions = []
        self.all_model2_completions = []

        # BERTScore
        self.bert_precision_model1 = []
        self.bert_precision_model2 = []
        self.bert_recall_model1 = []
        self.bert_recall_model2 = []
        self.bert_f1_model1 = []
        self.bert_f1_model2 = []

        # ROUGE-L
        self.rougeL_precision_model1 = []
        self.rougeL_precision_model2 = []
        self.rougeL_recall_model1 = []
        self.rougeL_recall_model2 = []
        self.rougeL_f1_model1 = []
        self.rougeL_f1_model2 = []


    def _summary_df(self, metrics):
        summary_rows = []
        for metric_name, values in metrics.items():
            arr = np.array(values, dtype=float)
            summary_rows.append(
                {
                    "metric": metric_name,
                    "mean": np.mean(arr) if arr.size else np.nan,
                    "std": np.std(arr) if arr.size else np.nan,
                }
            )
        return pd.DataFrame(summary_rows)

    def compare(self, model_dir1, model_dir2):
        bert_precision_model1 = []
        bert_precision_model2 = []
        bert_recall_model1 = []
        bert_recall_model2 = []
        bert_f1_model1 = []
        bert_f1_model2 = []

        rougeL_precision_model1 = []
        rougeL_precision_model2 = []
        rougeL_recall_model1 = []
        rougeL_recall_model2 = []
        rougeL_f1_model1 = []
        rougeL_f1_model2 = []

        model_1, tokenizer1 = load(model_dir1)
        model_2, tokenizer2 = load(model_dir2)

        for i,set in enumerate(tqdm(self.dataset)):

            messages_1 = [
                {"role": "system", "content": sys_prompt_model1},
                {"role": "user", "content": set['prompt']},
            ]

            messages2 = [
                #{"role": "system", "content": "Reasoning:\n"},
                {"role": "user", "content": set['prompt']},
            ]
            
            
            prompt1 = tokenizer1.apply_chat_template(
                messages_1, add_generation_prompt=True
            )


            prompt2 = tokenizer2.apply_chat_template(
                messages2, add_generation_prompt=True
            )
            

            try:
                from mlx_lm.sample_utils import make_sampler
                text1 = generate(
                    model_1,
                    tokenizer1,
                    prompt=prompt1,
                    verbose=False,
                    max_tokens=5000,
                    sampler=make_sampler(temp=0.6),
                )
                
                text2 = generate(
                    model_2,
                    tokenizer2,
                    prompt=prompt2,
                    verbose=False,
                    max_tokens=5000,
                    sampler=make_sampler(temp=0.6),
                )
                
                orig_text2 = copy.deepcopy(text2)
            
                text2 = text2.split("Solution:\n", 1)[1]

                completion = set['completion'].split("Solution:\n")[1]

                bert_p1, bert_r1, bert_f1_1 = score([text1], [completion], model_type=self.model_type, device='cpu')
                bert_p2, bert_r2, bert_f1_2 = score([text2], [completion], model_type=self.model_type, device='cpu')
                
                bert_precision_model1.append(bert_p1.item())
                bert_precision_model2.append(bert_p2.item())
                bert_recall_model1.append(bert_r1.item())
                bert_recall_model2.append(bert_r2.item())
                bert_f1_model1.append(bert_f1_1.item())
                bert_f1_model2.append(bert_f1_2.item())

                self.all_prompts.append(set['prompt'])
                self.all_ground_truth_completions.append(completion)
                self.all_model1_completions.append(text1)
                self.all_model2_completions.append(text2)


                r_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                rouge1 = r_scorer.score(completion, text1)['rougeL']
                rouge2 = r_scorer.score(completion, text2)['rougeL']

                rougeL_precision_model1.append(rouge1.precision)
                rougeL_precision_model2.append(rouge2.precision)
                rougeL_recall_model1.append(rouge1.recall)
                rougeL_recall_model2.append(rouge2.recall)
                rougeL_f1_model1.append(rouge1.fmeasure)
                rougeL_f1_model2.append(rouge2.fmeasure)
                
                print(self._summary_df(progress_metrics).to_string())

            except Exception as e:
                print(f"Failed on sample {i}: {e}")
            
            if i % 10 == 0:
                progress_metrics = {
                    "bert_precision_model1": bert_precision_model1,
                    "bert_precision_model2": bert_precision_model2,
                    "bert_recall_model1": bert_recall_model1,
                    "bert_recall_model2": bert_recall_model2,
                    "bert_f1_model1": bert_f1_model1,
                    "bert_f1_model2": bert_f1_model2,
                    "rougeL_precision_model1": rougeL_precision_model1,
                    "rougeL_precision_model2": rougeL_precision_model2,
                    "rougeL_recall_model1": rougeL_recall_model1,
                    "rougeL_recall_model2": rougeL_recall_model2,
                    "rougeL_f1_model1": rougeL_f1_model1,
                    "rougeL_f1_model2": rougeL_f1_model2,
                }
                print(f"\nProgress summary at sample {i}")
                print(self._summary_df(progress_metrics).to_string())


        final_metrics = {
            "bert_precision_model1": bert_precision_model1,
            "bert_precision_model2": bert_precision_model2,
            "bert_recall_model1": bert_recall_model1,
            "bert_recall_model2": bert_recall_model2,
            "bert_f1_model1": bert_f1_model1,
            "bert_f1_model2": bert_f1_model2,
            "rougeL_precision_model1": rougeL_precision_model1,
            "rougeL_precision_model2": rougeL_precision_model2,
            "rougeL_recall_model1": rougeL_recall_model1,
            "rougeL_recall_model2": rougeL_recall_model2,
            "rougeL_f1_model1": rougeL_f1_model1,
            "rougeL_f1_model2": rougeL_f1_model2,
        }

        print("\nFinal summary")
        print(self._summary_df(progress_metrics).to_string())

        self.bert_precision_model1 = bert_precision_model1
        self.bert_precision_model2 = bert_precision_model2
        self.bert_recall_model1 = bert_recall_model1
        self.bert_recall_model2 = bert_recall_model2
        self.bert_f1_model1 = bert_f1_model1
        self.bert_f1_model2 = bert_f1_model2

        self.rougeL_precision_model1 = rougeL_precision_model1
        self.rougeL_precision_model2 = rougeL_precision_model2
        self.rougeL_recall_model1 = rougeL_recall_model1
        self.rougeL_recall_model2 = rougeL_recall_model2
        self.rougeL_f1_model1 = rougeL_f1_model1
        self.rougeL_f1_model2 = rougeL_f1_model2

    def save_results(self):
        if len(self.bert_f1_model1) == 0 or len(self.bert_f1_model2) == 0:
            print("No results to save, please run the compare method first")
            return
        df = pd.DataFrame()
        df['prompt'] = self.all_prompts
        df['ground_truth_completion'] = self.all_ground_truth_completions
        df['model1_completion'] = self.all_model1_completions
        df['model2_completion'] = self.all_model2_completions
        df['bert_precision_model1'] = self.bert_precision_model1
        df['bert_precision_model2'] = self.bert_precision_model2
        df['bert_recall_model1'] = self.bert_recall_model1
        df['bert_recall_model2'] = self.bert_recall_model2
        df['bert_f1_model1'] = self.bert_f1_model1
        df['bert_f1_model2'] = self.bert_f1_model2
        df['rougeL_precision_model1'] = self.rougeL_precision_model1
        df['rougeL_precision_model2'] = self.rougeL_precision_model2
        df['rougeL_recall_model1'] = self.rougeL_recall_model1
        df['rougeL_recall_model2'] = self.rougeL_recall_model2
        df['rougeL_f1_model1'] = self.rougeL_f1_model1
        df['rougeL_f1_model2'] = self.rougeL_f1_model2

        _results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(_results_dir, exist_ok=True)
        df.to_csv(os.path.join(_results_dir, f"semantics_compare_to_pe_cot_{...}.csv"), index=False)

        return True
    


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        default="datasets/current_to_run/valid_IMPUTED.jsonl",
    )
    parser.add_argument(
        "--model_1",
        default="~/.lmstudio/models/personal/8b_noLora",
    )
    parser.add_argument(
        "--model_2",
        default="~/.lmstudio/models/submission/fuse_model_8b_qlora_manual_NEW_prompt",
    )
    args = parser.parse_args()

    dataset_dir = os.path.expanduser(args.dataset_dir)
    m1 = os.path.expanduser(args.model_1)
    m2 = os.path.expanduser(args.model_2)

    mc = ModelComparatorSemantics(dataset_dir=dataset_dir)
    mc.compare(m1, m2)
    mc.save_results()

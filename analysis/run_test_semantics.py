## Compare vanilla LLM to the fine-tuned LLM

import warnings
import numpy as np
import pandas as pd
import mlx.core as mx
from rouge_score import rouge_scorer
import re
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*") #annoying warnings
warnings.filterwarnings("ignore", message=".*interactive.*")
warnings.filterwarnings("ignore", category=UserWarning)
from transformers import logging
logging.set_verbosity_error()
logging.disable_progress_bar()
logging.set_verbosity(logging.CRITICAL)


from mlx_lm import load, generate
from tqdm import tqdm
from bert_score import score
import json

mx.random.seed(42)


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
                {"role": "user", "content": set['prompt']}
            ]

            messages2 = [
                {"role": "system", "content": "Reasoning:\n"}, # to fit the training template
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
                    tokenizer2,
                    prompt=prompt1,
                    verbose=False,
                    max_tokens=5000,
                    sampler=make_sampler(temp=0.7),
                )
                text2 = generate(
                    model_2,
                    tokenizer2,
                    prompt=prompt2,
                    verbose=False,
                    max_tokens=5000,
                    sampler=make_sampler(temp=0.7),
                )

                try:
                    text2 = text2.split("Solution:\n", 1)[1]
                except Exception: # Alternatively we can later try jsut adding the prompt to reason then solution as this worked fine in LMStudio
                    if isinstance(prompt2, str):
                        prompt2_text = prompt2
                    else:
                        prompt2_text = tokenizer2.decode(prompt2)
                    recovery_prompt = prompt2_text + text2.rstrip() + "\n\nSolution:\n"
                    recovery_completion = generate(
                        model_2,
                        tokenizer2,
                        prompt=recovery_prompt,
                        verbose=False,
                        max_tokens=1000,
                        sampler=make_sampler(temp=0.7),
                    )
                    recovered_text2 = text2.rstrip() + "\n\nSolution:\n" + recovery_completion.lstrip()
                    text2 = recovered_text2.split("Solution:\n", 1)[1]
                #text1 = text1.split("Solution:\n")[1]
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
                
                print(self._summary_df(progress_metrics).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

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
                print(self._summary_df(progress_metrics).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


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
        print(self._summary_df(final_metrics).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

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

        ###df.to_csv("analysis/results/semantics_comparison.csv")

        return True
    



if __name__ == "__main__":
    mc = ModelComparatorSemantics(dataset_dir = "/Users/michaelmurray/Documents/GitHub/RPMChem/datasets/current_to_run/valid_IMPUTED.jsonl")
    m1 = "/Users/michaelmurray/.lmstudio/models/personal/8b_noLora"
    m2 = "/Users/michaelmurray/.lmstudio/models/personal/fuse_model_8b_qlora_manual_NEW"
    mc.compare(m1,m2)
    mc.save_results()



"""
Just the code here
Final summary
                 metric   mean    std
  bert_precision_model1 0.5088 0.0967
  bert_precision_model2 0.6297 0.1216
     bert_recall_model1 0.6238 0.0863
     bert_recall_model2 0.5718 0.1366
         bert_f1_model1 0.5565 0.0846
         bert_f1_model2 0.5921 0.1174
rougeL_precision_model1 0.0993 0.0859
rougeL_precision_model2 0.3835 0.2857
   rougeL_recall_model1 0.4913 0.2255
   rougeL_recall_model2 0.2384 0.2351
       rougeL_f1_model1 0.1428 0.1015
       rougeL_f1_model2 0.2107 0.1839
"""















""" Skipped 80 but 122/288 with thresh - getting rid of thresh next (set to like np.inf) and running overnight
#WAS 0.15 THRESH
]                 metric   mean    std
  bert_precision_model1 0.5133 0.0893
  bert_precision_model2 0.6138 0.0931
     bert_recall_model1 0.6335 0.0707
     bert_recall_model2 0.6069 0.0694
         bert_f1_model1 0.5634 0.0768
         bert_f1_model2 0.6051 0.0626
rougeL_precision_model1 0.0982 0.0631
rougeL_precision_model2 0.2945 0.2168
   rougeL_recall_model1 0.5076 0.1935
   rougeL_recall_model2 0.2605 0.1820
       rougeL_f1_model1 0.1501 0.0815
       rougeL_f1_model2 0.2084 0.1047
"""

""" FULL RUN WITHOUT THRESHHOLDING
                 metric   mean    std
  bert_precision_model1 0.5335 0.0826
  bert_precision_model2 0.6245 0.1082
     bert_recall_model1 0.6247 0.0812
     bert_recall_model2 0.5846 0.1146
         bert_f1_model1 0.5718 0.0738
         bert_f1_model2 0.5972 0.0972
rougeL_precision_model1 0.1143 0.0741
rougeL_precision_model2 0.3569 0.2435
   rougeL_recall_model1 0.4647 0.1954
   rougeL_recall_model2 0.2587 0.2038
       rougeL_f1_model1 0.1635 0.0849
       rougeL_f1_model2 0.2227 0.1516
"""


""" FULL RUN WITHOUT THRESH BUT GIVING REASONING PROMPT TO VANILLA MODEL (NO REASONING PROMPT TO TRAINED MODEL)
                metric   mean    std
  bert_precision_model1 0.5896 0.0953
  bert_precision_model2 0.6251 0.0941
     bert_recall_model1 0.6104 0.0940
     bert_recall_model2 0.5830 0.1091
         bert_f1_model1 0.5950 0.0821
         bert_f1_model2 0.5970 0.0874
rougeL_precision_model1 0.2403 0.1901
rougeL_precision_model2 0.3508 0.2281
   rougeL_recall_model1 0.3336 0.2045
   rougeL_recall_model2 0.2561 0.2009
       rougeL_f1_model1 0.2148 0.1171
       rougeL_f1_model2 0.2200 0.1270
"""


""" FULL RUN WITHOUT THRESH WHERE BOTH MODELS GET THE FULL REASONING PROMPT
Final summary
                 metric   mean    std
  bert_precision_model1 0.5877 0.1018
  bert_precision_model2 0.6222 0.1006
     bert_recall_model1 0.6097 0.0940
     bert_recall_model2 0.5817 0.1108
         bert_f1_model1 0.5938 0.0862
         bert_f1_model2 0.5952 0.0940
rougeL_precision_model1 0.2413 0.1960
rougeL_precision_model2 0.3446 0.2343
   rougeL_recall_model1 0.3345 0.2076
   rougeL_recall_model2 0.2490 0.1913
       rougeL_f1_model1 0.2133 0.1227
       rougeL_f1_model2 0.2222 0.1443
"""
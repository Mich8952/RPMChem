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

'''
sys_promt_van = """
You provide solution manual-style answers to textbook questions.
Use this exact output format: "Reasoning:\n<your reasoning>\n\nSolution:\n<your final answer>".
1. Reasoning:
Think step by step about the problem and how to solve it
2. Solution:
Provide a textbook solution manual-style answer.

"""
'''


sys_prompt_model1 = """
You are a textbook solution solver.
Use this exact output format: "reasoning:\n<your reasoning>\n\nsolution:\n<your final answer>".
First start with "reasoning:" and discuss your solution approach.
Then provide "solution" in a textbook solution manual style.

Here are some few shot examples;

Question: '[d/dx, x^2] f(x) = d/dx (x^2 f(x)) - x^2 d/dx f(x)\n= 2x f(x) + x^2 d/dx f(x) - x^2 d/dx f(x) = 2x f(x)\n[d/dx, x^2] = 2x'
Answer: "Reasoning:\n<|channel|>analysis<|message|>We need to produce reasoning steps leading to commutator evaluation. Provide step-by-step: definition of commutator, apply to f(x), compute derivative product rule, subtract term, simplify, get 2x f(x). Then note operator equality.\n\nMake sure not to restate final answer verbatim? Actually we can present reasoning that leads to it but not a separate \"final answer\". So just steps. Ensure equations explicit. No markdown fences. Just plain text lines.\n\nLet's craft.<|end|><|start|>assistant<|channel|>final<|message|>The commutator of two operators \\(A\\) and \\(B\\) is defined by  \n\n\\[\n[A,B]\\,f(x)=A\\bigl(B f(x)\\bigr)-B\\bigl(A f(x)\\bigr).\n\\]\n\nTake \\(A=\\dfrac{d}{dx}\\) and \\(B=x^{2}\\).  Acting on an arbitrary function \\(f(x)\\),\n\n\\[\n[d/dx,\\,x^{2}]\\,f(x)=\\frac{d}{dx}\\!\\bigl(x^{2} f(x)\\bigr)-x^{2}\\,\\frac{d}{dx}f(x).\n\\]\n\nUse the product rule for the first term:\n\n\\[\n\\frac{d}{dx}\\!\\bigl(x^{2} f(x)\\bigr)=\\frac{d}{dx}(x^{2})\\,f(x)+x^{2}\\,\\frac{d}{dx}f(x)\n=2x\\,f(x)+x^{2}\\,f'(x).\n\\]\n\nSubstitute this result back into the commutator expression:\n\n\\[\n[d/dx,\\,x^{2}]\\,f(x)=\\bigl[2x\\,f(x)+x^{2}f'(x)\\bigr]-x^{2}f'(x).\n\\]\n\nThe \\(x^{2}f'(x)\\) terms cancel, leaving\n\n\\[\n[d/dx,\\,x^{2}]\\,f(x)=2x\\,f(x).\n\\]\n\nSince this holds for any function \\(f(x)\\), the operator identity is  \n\n\\[\n[d/dx,\\,x^{2}]=2x.\n\\]\n\nSolution:\n[d/dx, x^2] f(x) = d/dx (x^2 f(x)) - x^2 d/dx f(x)\n= 2x f(x) + x^2 d/dx f(x) - x^2 d/dx f(x) = 2x f(x)\n[d/dx, x^2] = 2x"


Question: 'Use the information in Exercise E13E.3(b) to calculate the electronic contribution to the molar Gibbs energy of O2 at 400 K.'
Answer: 'Reasoning:\n<|channel|>analysis<|message|>We need to produce reasoning steps leading to that solution, not restating final answer. Provide step-by-step derivation: relation G = -RT ln q^E for molar electronic contribution; compute degeneracies from J values; compute βhcν term; evaluate exponent at given temperatures (400 K? Wait question says 400 K but solution uses 500 and 900 K—maybe they used those as examples). But we follow provided final solution. Provide steps: partition function factorization, expression for q^E, calculate degeneracies, compute energy gap in temperature units, plug into exponent, evaluate q^E at each T, then G^E_m = -RT ln q^E.\n\nWe must be concise but complete, include unit conversions and constants.\n\nLet's write reasoning steps.<|end|><|start|>assistant<|channel|>final<|message|>1. Write the molar Gibbs‑energy contribution of a subsystem in terms of its partition function  \n\n\\[\nG^{E}_{m}= -RT\\ln q^{E}\n\\]\n\nbecause the factor \\(1/N\\) that appears in the full expression for \\(G\\) is already taken into account by the translational part; only the electronic part remains as \\(-RT\\ln q^{E}\\).\n\n2. For a diatomic molecule whose lowest two electronic levels are separated by an energy \\(\\Delta E = hc\\tilde{\\nu}\\), the electronic partition function of this two‑level system is  \n\n\\[\nq^{E}=g_{0}+g_{1}\\,e^{-\\beta hc\\tilde{\\nu}},\\qquad \n\\beta=\\frac{1}{k_{\\!B}T}\n\\]\n\nwhere \\(g_{i}=2J_{i}+1\\) are the degeneracies of the levels.\n\n3. Determine the degeneracies from the given total‑angular‑momentum quantum numbers  \n\n\\[\nJ_{0}= \\tfrac32 \\;\\Rightarrow\\; g_{0}=2J_{0}+1 = 4\n\\]\n\\[\nJ_{1}= \\tfrac12 \\;\\Rightarrow\\; g_{1}=2J_{1}+1 = 2\n\\]\n\nThus  \n\n\\[\nq^{E}=4+2\\,e^{-\\beta hc\\tilde{\\nu}} .\n\\]\n\n4. Convert the energy gap to a temperature factor. Using  \n\n\\[\nhc\\tilde{\\nu}= (6.6261\\times10^{-34}\\,\\text{J·s})(2.9979\\times10^{10}\\,\\text{cm s}^{-1})(881\\,\\text{cm}^{-1})\n\\]\n\ngives  \n\n\\[\nhc\\tilde{\\nu}= 1.74\\times10^{-20}\\,\\text{J}.\n\\]\n\nDivide by Boltzmann’s constant to obtain the equivalent temperature  \n\n\\[\n\\frac{hc\\tilde{\\nu}}{k_{\\!B}}=\n\\frac{1.74\\times10^{-20}\\,\\text{J}}\n     {1.3806\\times10^{-23}\\,\\text{J K}^{-1}}\n= 1.26\\times10^{3}\\,\\text{K}.\n\\]\n\nHence  \n\n\\[\n\\beta hc\\tilde{\\nu}= \\frac{hc\\tilde{\\nu}}{k_{\\!B}T}\n                    =\\frac{1.26\\times10^{3}\\,\\text{K}}{T}.\n\\]\n\n5. Evaluate the exponential factor at each temperature.\n\n*At \\(T=500\\;\\text{K}\\):*  \n\n\\[\ne^{-\\beta hc\\tilde{\\nu}}=\ne^{- (1.26\\times10^{3}/500)} = e^{-2.52}=0.080.\n\\]\n\nThus  \n\n\\[\nq^{E}(500)=4+2(0.080)=4.16 .\n\\]\n\n*At \\(T=900\\;\\text{K}\\):*  \n\n\\[\ne^{-\\beta hc\\tilde{\\nu}}=\ne^{- (1.26\\times10^{3}/900)} = e^{-1.40}=0.247.\n\\]\n\nThus  \n\n\\[\nq^{E}(900)=4+2(0.247)=4.49 .\n\\]\n\n6. Insert the partition‑function values into the Gibbs‑energy expression.\n\n*For \\(T=500\\;\\text{K}\\):*  \n\n\\[\nG^{E}_{m}= - (8.3145\\,\\text{J mol}^{-1}\\,\\text{K}^{-1})(500\\,\\text{K})\\,\n           \\ln(4.16)\n         = -5.9\\times10^{3}\\,\\text{J mol}^{-1}\n         = -5.9\\;\\text{kJ \n\nSolution:\nThe Gibbs energy is computed from the partition function using [13F.8-568], G(T) = G(0) - nRT ln q/N. As usual, the partition function is factored into separate contributions from translation, rotation and so on. The factor of 1/N is usually taken with the translational contribution, therefore the electronic contribution to the Gibbs energy is -nRT ln q^E, or -RT ln q^E for the molar quantity.\n\nThe electronic partition function of this two-level system is\n\nq^E = g_0 + g_1 e^{-\\beta hc\\tilde{\\nu}}\n\nwhere g_0 and g_1 are the degeneracies of the ground and first excited state, respectively. For an electronic term the degeneracy is 2J + 1, hence g_0 = 2 × 3/2 + 1 = 4 and g_1 = 2 × 1/2 + 1 = 2. With the data given\n\n\\beta hc\\tilde{\\nu} = hc\\tilde{\\nu}/k × T^{-1} = (6.6261 × 10^{-34} J s) × (2.9979 × 10^{10} cm s^{-1}) × (881 cm^{-1}) / (1.3806 × 10^{-23} J K^{-1}) × T^{-1} = (1.26... × 10^3 K) × T^{-1}\n\nat 500 K\n\nq^E = 4 + 2e^{-(1.26...×10^3 K)/(500 K)} = 4.15...\n\nG^E_m = -RT ln q^E = -(8.3145 J K^{-1} mol^{-1}) × (500 K) × ln(4.15...) = -5.92 kJ mol^{-1}\n\nat 900 K\n\nq^E = 4 + 2e^{-(1.26...×10^3 K)/(900 K)} = 4.48...\n\nG^E_m = -(8.3145 J K^{-1} mol^{-1}) × (900 K) × ln(4.48...) = -11.2 kJ mol^{-1}

Question: "In this problem you will establish the commutation relations, given in eqn 7E.14, between the operators for the x-, y-, and z-components of angular momentum, which are defined in eqn 7F.13. In order to manipulate the operators correctly it is helpful to imagine that they are acting on some arbitrary function f; it does not matter what f is. Consider the effect of the first term on some arbitrary function f and evaluate [l_x, l_y]f = l_x l_y f - l_y l_x f. The next step is to multiply out the parentheses, and in doing so care needs to be taken over the order of operations. (b) Repeat the procedure for the other term in the commutator; l_y l_x f. (c) Combine the results from (a) and (b) to evaluate l_x l_y f - l_y l_x f; you should find that many of the terms cancel. Confirm that the final expression you have is indeed iħl_z f, where l_z is given in eqn 7F.13. The definitions in eqn 7F.13 are related to one another by"
Answer: "Reasoning:\n<|channel|>analysis<|message|>We need to produce reasoning steps leading to the given solution, not restate final answer. So we outline derivation: define operators l_x = (ħ/i)(y∂/∂z - z∂/∂y), etc. Compute l_x l_y f by applying operator sequentially, use product rule, expand terms, similarly for l_y l_x f, then subtract, cancel second derivative terms due to symmetry, left with first-derivative combination giving iħ l_z f. Also note cyclic permutations give other commutators.\n\nWe must keep equations explicit and concise. Provide reasoning steps (a)-(e). No final answer section. No markdown fences. Just plain text.\n\nLet's craft.<|end|><|start|>assistant<|channel|>final<|message|>**(a)  Action of \\( \\hat l_x\\hat l_y\\) on an arbitrary function**\n\nWrite the operators in differential form  \n\n\\[\n\\hat l_x=\\frac{\\hbar}{i}\\Bigl(y\\frac{\\partial}{\\partial z}-z\\frac{\\partial}{\\partial y}\\Bigr),\\qquad   \n\\hat l_y=\\frac{\\hbar}{i}\\Bigl(z\\frac{\\partial}{\\partial x}-x\\frac{\\partial}{\\partial z}\\Bigr).\n\\]\n\nHence  \n\n\\[\n\\hat l_x\\hat l_y f\n =-\\hbar^{2}\\Bigl(y\\frac{\\partial}{\\partial z}-z\\frac{\\partial}{\\partial y}\\Bigr)\n        \\Bigl(z\\,\\frac{\\partial f}{\\partial x}-x\\,\\frac{\\partial f}{\\partial z}\\Bigr).\n\\]\n\nDistribute the outer brackets, remembering that the derivatives act on everything to their right:\n\n\\[\n\\begin{aligned}\n\\hat l_x\\hat l_y f\n=&-\\hbar^{2}\\Bigl[ \n      y\\frac{\\partial}{\\partial z}\\!\\bigl(z\\,\\partial_{x}f\\bigr)\n    - y\\frac{\\partial}{\\partial z}\\!\\bigl(x\\,\\partial_{z}f\\bigr)   \\\\\n &\\qquad\n    - z\\frac{\\partial}{\\partial y}\\!\\bigl(z\\,\\partial_{x}f\\bigr)\n    + z\\frac{\\partial}{\\partial y}\\!\\bigl(x\\,\\partial_{z}f\\bigr)\\Bigr].\n\\end{aligned}\n\\]\n\nApply the product rule to each term.  For example  \n\n\\[\ny\\frac{\\partial}{\\partial z}\\!\\bigl(z\\,\\partial_{x}f\\bigr)\n   = y\\Bigl(\\frac{\\partial z}{\\partial z}\\,\\partial_{x}f\n            +z\\,\\partial^{2}_{zx}f\\Bigr)\n   = y\\,\\partial_{x}f+yz\\,\\partial^{2}_{zx}f .\n\\]\n\nCarrying this out for all four pieces gives  \n\n\\[\n\\hat l_x\\hat l_y f=\n-\\hbar^{2}\\Bigl[ \n      y\\,\\partial_{x}f\n    + yz\\,\\partial^{2}_{zx}f\n    - xy\\,\\partial^{2}_{zz}f\n    - z^{2}\\,\\partial^{2}_{yx}f\n    + zx\\,\\partial^{2}_{yz}f\\Bigr].\n\\]\n\n---\n\n**(b)  Action of \\( \\hat l_y\\hat l_x\\) on the same function**\n\nSwap the order of the operators:\n\n\\[\n\\hat l_y\\hat l_x f\n =-\\hbar^{2}\\Bigl(z\\frac{\\partial}{\\partial x}-x\\frac{\\partial}{\\partial z}\\Bigr)\n        \\Bigl(y\\,\\partial_{z}f - z\\,\\partial_{y}f\\Bigr).\n\\]\n\nDistribute and use the product rule exactly as above:\n\n\\[\n\\begin{aligned}\n\\hat l_y\\hat l_x f\n=&-\\hbar^{2}\\Bigl[ \n      z\\frac{\\partial}{\\partial x}(y\\,\\partial_{z}f)\n    - z\\frac{\\partial}{\\partial x}(z\\,\\partial_{y}f)   \\\\\n &\\qquad\n    - x\\frac{\\partial}{\\partial z}(y\\,\\partial_{z}f)\n    + x\\frac{\\partial}{\\partial z}(z\\,\\partial_{y}f)\\Bigr]  \\\\[2mm]\n=&-\\hbar^{2}\\Bigl[\n      yz\\,\\partial^{2}_{xz}f\n    - z^{2}\\,\\partial^{2}_{xy}f\n    - xy\\,\\partial^{2\n\nSolution:\n(a) Multiplying out the brackets, noting that the derivatives in the left brackets act on the whole term in the right brackets, e.g. x∂f/∂x\n\nl̂x l̂y f = −ħ² (y ∂/∂z − z ∂/∂y) (z ∂f/∂x − x ∂f/∂z)\n= −ħ² [y ∂/∂z (z ∂f/∂x) − y ∂/∂z (x ∂f/∂z) − z ∂/∂y (z ∂f/∂x) + z ∂/∂y (x ∂f/∂z)]\nTo evaluate the first term in this, the product rule is used\n= −ħ² [y ∂z/∂z ∂f/∂x + yz ∂²f/∂z∂x − x y ∂²f/∂z² − z² ∂²f/∂y∂x + z x ∂²f/∂y∂z]\n= −ħ² [y ∂f/∂x + yz ∂²f/∂z∂x − x y ∂²f/∂z² − z² ∂²f/∂y∂x + z x ∂²f/∂y∂z]\n\n(b) Similarly,\n\nl̂y l̂x f = −ħ² (z ∂/∂x − x ∂/∂z) (y ∂f/∂z − z ∂f/∂y)\n= −ħ² [z ∂/∂x (y ∂f/∂z) − z ∂/∂x (z ∂f/∂y) − x ∂/∂z (y ∂f/∂z) + x ∂/∂z (z ∂f/∂y)]\nTo evaluate the final term in this, the product rule must be used\n= −ħ² [y z ∂²f/∂x∂z − z² ∂²f/∂x∂y − x y ∂²f/∂z² + x z ∂²f/∂z∂y]\n= −ħ² [y z ∂²f/∂x∂z − z² ∂²f/∂x∂y − x y ∂²f/∂z² + x z ∂²f/∂z∂y]\n\n(c) Due to the symmetry of mixed partial derivatives, the only terms that are not repeated in both of these terms are the first derivatives. Hence,\n\nl̂x l̂y f − l̂y l̂x f = ħ² (x ∂f/∂y − y ∂f/∂x) = iħ × ħ/i (x ∂f/∂y − y ∂f/∂x) = iħ l̂z f\nwhere the definition of the l̂z operator given in [7F.13–289] is used. It follows that [l̂x, l̂y] = l̂z.\n\n(d) Applying cyclic permutation to l̂x gives (ħ/i)(z∂/∂x − x∂/∂z) = l̂y. Likewise l̂y gives (ħ/i)(x∂/∂y − y∂/∂x) = l̂z, and l̂z gives (ħ/i)(y∂/∂z − z∂/∂y) = l̂x.\n\n(e) Applying this permutation to the expression [l̂x, l̂y] = iħ l̂z gives [l̂y, l̂z] = iħ l̂x. Permuting this expression gives [l̂z, l̂x] = iħ l̂y, and permuting that expression gives [l̂x, l̂y] = iħ l̂z."}

"""


class ModelComparatorSemantics:
    def __init__(self, dataset_dir = "datasets/processed/valid.jsonl", model_type = "allenai/scibert_scivocab_uncased"):
        with open(dataset_dir, "r") as f:
            self.dataset = [json.loads(line) for line in f]
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

    def _equation_heavy_score(self, text):
        if not isinstance(text, str) or len(text) == 0:
            return 0.0

        length = len(text)
        equation_chars = sum(ch in "=+-*/^()[]{}<>|\\" for ch in text)
        digits = sum(ch.isdigit() for ch in text)
        latex_like = len(re.findall(r"\\[a-zA-Z]+|\$[^\$]+\$|\^\{|_\{|\{[^\}]*\}", text))
        eq_lines = len(re.findall(r"(?m)^\s*[A-Za-z][A-Za-z0-9_]*\s*=", text))

        score = (
            (equation_chars / length)
            + 0.45 * (digits / length)
            + 0.03 * latex_like
            + 0.05 * eq_lines
        )
        return float(score)

    def _is_equation_heavy(self, text):
        if not isinstance(text, str):
            return False
        return self._equation_heavy_score(text) >= self.equation_heavy_threshold

    def _extract_solution(self, text):
        if not isinstance(text, str):
            raise ValueError("Completion is not a string")
        match = re.search(r"(?is)\bsolution\s*:\s*", text)
        if match is None:
            raise ValueError("Missing solution section")
        return text[match.end():]
    
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


""" 209/403
                 metric   mean    std
  bert_precision_model1 0.5031 0.0935
  bert_precision_model2 0.6418 0.1066
     bert_recall_model1 0.6182 0.0745
     bert_recall_model2 0.5751 0.1403
         bert_f1_model1 0.5512 0.0790
         bert_f1_model2 0.5991 0.1144
rougeL_precision_model1 0.0958 0.0796
rougeL_precision_model2 0.3987 0.2733
   rougeL_recall_model1 0.4928 0.2056
   rougeL_recall_model2 0.2497 0.2369
       rougeL_f1_model1 0.1422 0.0955
       rougeL_f1_model2 0.2285 0.1888

       
"""
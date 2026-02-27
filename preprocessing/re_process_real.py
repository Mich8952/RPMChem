import json
from tqdm import tqdm   
import re
import numpy as np

class ReprocessorReal:
    def __init__(self, dir_to_file):
        with open(dir_to_file, "r") as f:
            data = [json.loads(line) for line in f]
        self.datestamp = dir_to_file.split("/")[-1].split(".")[0].split("full_dataset_")[-1]
        self.data = data

        self.blacklist = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","a)","b)","c)","d)","e)","f)","g)","h)","i)","j)","k)","l)","m)","n)","o)","p)","q)","r)","s)","t)","u)","v)","w)","x)","y)","z)","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A)","B)","C)","D)","E)","F)","G)","H)","I)","J)","K)","L)","M)","N)","O)","P)","Q)","R)","S)","T)","U)","V)","W)","X)","Y)","Z)"]

        self.pattern = re.compile(r'^[PQ]\d+(?:\.\d+)?$')
        self.disallowed_phrases = ["refer to", "figure", "table", "see question"]

        self.refined_data = []

    
    def clean_jsons(self):
        refined_data = []
        for i,d in enumerate(tqdm(self.data)):    
            refined_data_singlet = {}
            prompt_lower = d.get("prompt", "").lower()
            completion_lower = d.get("completion", "").lower()
            has_disallowed_phrase = any(
                phrase in prompt_lower or phrase in completion_lower
                for phrase in self.disallowed_phrases
            )

            if (
                (d['prompt'] not in self.blacklist) and ("completion" in d.keys())
                and (self.pattern.match(d['question_num']) is not None)
                and (d['completion'] != "")
                and (not has_disallowed_phrase)
            ):
                try:
                    refined_data_singlet['prompt'] = d['prompt']
                    refined_data_singlet['completion'] = d['completion']
                    refined_data.append(refined_data_singlet)
                except:
                    print(f"failed when processing sample {i}, plz check")

        self.refined_data = np.array(refined_data)

    def split_data(self,test_prop = 0.2):
        if self.refined_data is None:
            raise Exception("Could not find any refined_data object attribute. make sure you run self.clean_jsons")

        permuted_idx = np.random.permutation(len(self.refined_data))
        data_to_split = self.refined_data[permuted_idx]

        train_data = data_to_split[:int((1-test_prop)*len(self.refined_data))]
        test_data = data_to_split[int((1-test_prop)*len(self.refined_data)):]

        with open(f'datasets/current_to_run/train.jsonl', "w", encoding="utf-8") as f:  
            for d in train_data:
                f.write(json.dumps(d))
                f.write("\n")

        with open(f'datasets/current_to_run/valid.jsonl', "w", encoding="utf-8") as f:  
            for d in test_data:
                f.write(json.dumps(d))
                f.write("\n")
                
        print("Done processing")


        
if __name__ == "__main__":
    rp = ReprocessorReal("/Users/michaelmurray/Documents/GitHub/RPMChem/datasets/processed_real/joined_2026-02-20__08_01_44__301077.jsonl")
    rp.clean_jsons()
    rp.split_data()

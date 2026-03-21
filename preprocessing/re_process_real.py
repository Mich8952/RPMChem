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

        #self.pattern = re.compile(r'^[A-Z]\d{1,2}(?:[A-Z])?\.\d+(?:\([a-z]\))?$')
        self.disallowed_phrases = ["refer to", "figure", "table", "see question"]

        self.refined_data = []

    
    def clean_jsons(self):
        bs=0
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
                #and (self.pattern.match(d['question_num']) is not None)
                and (d['completion'] != "")
                and (not has_disallowed_phrase)
            ):
                try:
                    refined_data_singlet['prompt'] = d['prompt']
                    refined_data_singlet['completion'] = d['completion']
                    refined_data.append(refined_data_singlet)
                    txt_id = d.get("textbook_id")

                    if txt_id is not None:
                        refined_data_singlet['textbook_id'] = txt_id
                except:
                    print(f"failed when processing sample {i}, plz check")
            else:
                bs+=1
                print(f"Skipping bad sample {bs}")

        self.refined_data = np.array(refined_data)

    def split_data(self,test_prop = 0.15):
        if self.refined_data is None:
            raise Exception("Could not find any refined_data object attribute. make sure you run self.clean_jsons")

        permuted_idx = np.random.permutation(len(self.refined_data))
        data_to_split = self.refined_data[permuted_idx]

        train_data = data_to_split[:int((1-test_prop)*len(self.refined_data))]
        test_data = data_to_split[int((1-test_prop)*len(self.refined_data)):]

        train_string = f"datasets/e2e_artifacts/train_noimpute_{self.datestamp}.jsonl"
        test_string = f"datasets/e2e_artifacts/valid_noimpute_{self.datestamp}.jsonl"

        with open(train_string, "w", encoding="utf-8") as f:  
            for d in train_data:
                f.write(json.dumps(d))
                f.write("\n")

        with open(test_string, "w", encoding="utf-8") as f:  
            for d in test_data:
                f.write(json.dumps(d))
                f.write("\n")
                
        print("Done processing")

        return train_string, test_string


        
if __name__ == "__main__":
    rp = ReprocessorReal("datasets/processed_real/mega_joined_3txt_with_textbook_ids.jsonl")
    rp.clean_jsons()
    rp.split_data()

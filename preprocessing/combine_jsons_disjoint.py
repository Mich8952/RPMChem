import numpy as np
import pandas as pd
import json

class Combiner:
    def __init__(self, question_json_path, solution_json_path, output_json_path):
        self.question_json_path = question_json_path
        self.solution_json_path = solution_json_path
        self.output_json_path = output_json_path

    @staticmethod
    def load_and_convert_to_df(dir_json):
        with open(dir_json, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame()

        all_qnums = []
        all_qa = []
        for record in data['records']:
            all_qnums.append(record['question_num'])
            if 'question_text' in record:
                all_qa.append(record['question_text'])
            else:
                all_qa.append(record['answer_text'])
        
        
        df['question_num'] = all_qnums
        if 'question_text' in record:
            df['question_text'] = all_qa
        else:
            df['answer_text'] = all_qa
    

        return df
    
    def __call__(self):
        question_df = self.load_and_convert_to_df(self.question_json_path)
        solution_df = self.load_and_convert_to_df(self.solution_json_path)

        combined_df = pd.merge(question_df, solution_df, on='question_num')
        combined_df['valid'] = True

        combined_df.columns = ["question_num","prompt","completion","valid"]
        combined_df = combined_df[["valid","question_num","prompt","completion"]]

        with open(self.output_json_path, 'w') as f:
            combined_df.to_json(f, orient='records', lines=True)

        return self.output_json_path

        

if __name__ == "__main__":
    question_json_path = "/Users/michaelmurray/Documents/GitHub/RPMChem/datasets/processed_real/pdf1_json.json"
    solution_json_path = "/Users/michaelmurray/Documents/GitHub/RPMChem/datasets/processed_real/pdf2_json.json"
    output_json_path = "/Users/michaelmurray/Documents/GitHub/RPMChem/datasets/processed_real/joined_disjoint_textbook.jsonl"

    combiner = Combiner(question_json_path, solution_json_path, output_json_path)
    combiner()
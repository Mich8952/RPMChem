import json

class WrapperJsonP:
    def __init__(self, path_to_json):
        self.name = path_to_json.split("/")[-1].split('.jsonl')[0]
        self.rows = []
        with open(path_to_json,'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
    @staticmethod
    def write_out(out_dir, name, rows: list):
        file_path = f"{out_dir}/PWPJ_{name}.jsonl"

        with open(file_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
            
    def __call__(self):
        temp_rows = []
        for row in self.rows:
            temp_rows.append({"valid":True, "question_num":"na","prompt":row['prompt'],"completion":row['completion']})
        
        WrapperJsonP.write_out("datasets/processed_real",self.name, temp_rows)


if __name__ == "__main__":
    #dir = "/Users/michaelmurray/Documents/GitHub/RPMChem/datasets/paolas_jsons/ball_prompt_completion (1).jsonl"
    dir = "/Users/michaelmurray/Documents/GitHub/RPMChem/datasets/paolas_jsons/flower_prompt_completion.jsonl"

    WJP = WrapperJsonP(dir)
    WJP()
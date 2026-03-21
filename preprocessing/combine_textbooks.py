import json

class TextbookCombiner:
    def __init__(self, jsonl_paths):
        self.jsonl_paths = jsonl_paths

    def grab_txt_id(self, path):
        return path.split("_")[-2]
 
    def __call__(self, output_path):
        with open(output_path, "w", encoding="utf-8") as writer:
            for jsonl_path in self.jsonl_paths:
                textbook_id = self.grab_txt_id(jsonl_path)
                with open(jsonl_path, "r", encoding="utf-8") as reader:
                    for line in reader:
                        line = line.strip()
                        if not line:
                            continue
                        payload = json.loads(line)
                        payload["textbook_id"] = textbook_id
                        writer.write(json.dumps(payload))
                        writer.write("\n")

        return output_path

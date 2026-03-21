## Code to run e2e preprocessing
# 1. run get_jsons (joint/disjoint depends on the file)
# 2. run re_process_real (removes bad samples
# 3. (optional reasoning imputation) add_reasoning_context.py
import uuid
import time
from pathlib import Path
from preprocessing.get_jsons_disjoint_textbook import PdfExtractor, extract_items_from_pdf, should_scan_pdf1_page, prompt_pdf1_questions, prompt_pdf2_answers
from combine_jsons_disjoint import Combiner
from combine_textbooks import TextbookCombiner
from re_process_real import ReprocessorReal
from add_reasoning_context import SplitProcessor

class Preprocessor:
    def __init__(self, pdfs_to_process : list[tuple[str]], impute : bool = True):
        self.pdfs_to_process = pdfs_to_process # this is a list of tuples because
        self.session_id = str(uuid.uuid4())
        self.textbook_counter = 0
        self.impute = impute

    def proc_disjoint_txt(self, pdf_path_txt, pdf_path_solns):
        # this is for when the textbooks are disjoint (textbook and solns manual are separate)
        pdf1_json = f"datasets/e2e_artifacts/pdf1_json_from_pipeline_{self.textbook_counter}_{self.session_id}.json"
        pdf2_json = f"datasets/e2e_artifacts/pdf2_json_from_pipeline_{self.textbook_counter}_{self.session_id}.json"

        pdf1_text_extractor = PdfExtractor(pdf_path_txt)

        extract_items_from_pdf(
            pdf_path_txt,
            pdf1_json,
            prompt_pdf1_questions,
            should_scan_page=lambda page_num: should_scan_pdf1_page(page_num, pdf1_text_extractor),
        )
        extract_items_from_pdf(pdf_path_solns, pdf2_json, prompt_pdf2_answers)

        return pdf1_json, pdf2_json

    #def proc_joint_txt():
        #...#ASSUMPTION maybe we just pass tthe same pdf as source and as dest from proc_disjoint instead
    
    def combine_jsons_from_disjoint(self, pdf_q, pdf_a):
        combiner = Combiner(pdf_q, pdf_a, f"datasets/e2e_artifacts/joined_disjoint_{self.textbook_counter}_{self.session_id}.jsonl")
        return combiner() # returns path

    def __call__(self):
        joint_paths = []
        for pdf_pair in self.pdfs_to_process:
            pdf_q, pdf_a = self.proc_disjoint_txt(*pdf_pair)
            time.sleep(5)
            joint_path = self.combine_jsons_from_disjoint(pdf_q,pdf_a)
            joint_paths.append(joint_path)
            self.textbook_counter += 1 
            time.sleep(5)
        # now that you processed all textbooks, we just need to merge them into one jsonl
        tc = TextbookCombiner(joint_paths)
        combined_path = tc(f"datasets/e2e_artifacts/mega_joined_{self.session_id}.jsonl")
        time.sleep(5)

        # now reprocess to remove bad samples (this is fast and regex based)
        reprocessor = ReprocessorReal(combined_path)
        reprocessor.clean_jsons()
        train_string, test_string = reprocessor.split_data()

        #now impute
        #note that this call uses the Path functionliaty so we need to wrap strings as path objects
        if self.impute:
            sp = SplitProcessor()
            sp.process_split(Path(train_string), Path(train_string.replace(".jsonl", "_reasoning.jsonl")))
            sp.process_split(Path(test_string), Path(test_string.replace(".jsonl", "_reasoning.jsonl")))

        time.sleep(5)




if __name__ == "__main__":
    pdfs_to_process = [
        ("/Users/michaelmurray/Documents/GitHub/RPMChem/datasets/raw/Thomas Engel and Philip Reid - Solution Manual for Physical Chemistry (0).pdf" , "/Users/michaelmurray/Documents/GitHub/RPMChem/datasets/raw/Thomas Engel and Philip Reid - Solution Manual for Physical Chemistry (0).pdf"),
        
        ("/Users/michaelmurray/Documents/GitHub/RPMChem/datasets/pdfs/Atkins_ Physical Chemistry 11e.pdf", "/Users/michaelmurray/Documents/GitHub/RPMChem/datasets/pdfs/Solutions Manual - Atkins Physical Chemistry 11th Ed.pdf"),

        # I dont have the flower textbook, need to ask paola
    
    ]

    preprocessor = Preprocessor(pdfs_to_process)
    preprocessor()
# preprocessor_pipeline.py

"""
End-to-end preprocessing pipeline.

Steps:
1. Extract Q/A items from PDFs (get_jsons_joint or get_jsons_disjoint_textbook).
2. Remove bad samples (re_process_real).
3. Optional: impute reasoning context (add_reasoning_context).
"""


import argparse
import time
import uuid

import yaml

from add_reasoning_context import SplitProcessor
from combine_jsons_disjoint import Combiner
from combine_textbooks import TextbookCombiner
from extract_numerical_subset import NumberExtractor
from get_jsons_disjoint_textbook import (
    PdfExtractor,
    extract_items_from_pdf,
    should_scan_pdf1_page,
    build_disjoint_question_prompt,
    build_disjoint_answer_prompt,
)
from re_process_real import ReprocessorReal

INTER_STEP_SLEEP = 5


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Preprocessor:
    """
    Runs the full preprocessing pipeline for one or more textbook PDF pairs.

    Args:
        pdfs_to_process: List of (question_pdf_path, solutions_pdf_path) tuples.
            For joint PDFs, both elements of the tuple are the same path.
        impute: If True, augment training and validation splits with
            model-generated reasoning traces.
    """

    def __init__(
        self,
        pdfs_to_process: list[tuple[str, str]],
        impute: bool = True,
    ):
        self.pdfs_to_process = pdfs_to_process
        self.session_id = str(uuid.uuid4())
        self.textbook_counter = 0
        self.impute = impute

    def _artifact_path(self, name: str, ext: str) -> str:
        """Return a session-scoped path under datasets/e2e_artifacts/."""
        return (
            f"datasets/e2e_artifacts/"
            f"{name}_{self.textbook_counter}_{self.session_id}.{ext}"
        )

    def proc_disjoint_txt(
        self, pdf_path_txt: str, pdf_path_solns: str
    ) -> tuple[str, str]:
        """
        Extract questions and answers from a disjoint textbook/solutions pair.

        Returns:
            Paths to the question JSON and solutions JSON respectively.
        """
        pdf1_json = self._artifact_path("pdf1_json_from_pipeline", "json")
        pdf2_json = self._artifact_path("pdf2_json_from_pipeline", "json")

        pdf1_text_extractor = PdfExtractor(pdf_path_txt)

        extract_items_from_pdf(
            pdf_path_txt,
            pdf1_json,
            build_disjoint_question_prompt,
            final_page=pdf1_text_extractor.num_pages,
            should_scan_page=lambda page_num: should_scan_pdf1_page(
                page_num, pdf1_text_extractor
            ),
        )
        extract_items_from_pdf(
            pdf_path_solns,
            pdf2_json,
            build_disjoint_answer_prompt,
        )

        return pdf1_json, pdf2_json

    def combine_jsons_from_disjoint(
        self, pdf_q: str, pdf_a: str
    ) -> str:
        """Merge question and solution JSONs into a single JSONL file."""
        output_path = self._artifact_path("joined_disjoint", "jsonl")
        combiner = Combiner(pdf_q, pdf_a, output_path)
        return combiner()

    def __call__(self) -> None:
        """Execute the full pipeline end-to-end."""
        joint_paths = []

        for pdf_pair in self.pdfs_to_process:
            pdf_q, pdf_a = self.proc_disjoint_txt(*pdf_pair)
            time.sleep(INTER_STEP_SLEEP)
            joint_path = self.combine_jsons_from_disjoint(pdf_q, pdf_a)
            joint_paths.append(joint_path)
            self.textbook_counter += 1
            time.sleep(INTER_STEP_SLEEP)

        combined_path = TextbookCombiner(joint_paths)(
            f"datasets/e2e_artifacts/mega_joined_{self.session_id}.jsonl"
        )
        time.sleep(INTER_STEP_SLEEP)

        reprocessor = ReprocessorReal(combined_path)
        reprocessor.clean_jsons()
        train_path, test_path = reprocessor.split_data()

        if self.impute:
            sp = SplitProcessor()
            sp.process_split(
                train_path,
                train_path.replace(".jsonl", "_reasoning.jsonl"),
            )
            sp.process_split(
                test_path,
                test_path.replace(".jsonl", "_reasoning.jsonl"),
            )

        test_base = test_path.removesuffix(".jsonl")
        NumberExtractor(
            test_path,
            output_csv=f"{test_base}_numerical_prompts.csv",
        ).run_all()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run preprocessing pipeline from YAML config"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="conf/preprocessing.yaml",
    )
    parser.add_argument(
        "--no-impute",
        action="store_true",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    textbook_dir = config["textbook_dir"].rstrip("/")
    pdfs_to_process = []

    for textbook in config["textbooks"]:
        mode = textbook["mode"]
        if mode == "joint":
            q_path = f"{textbook_dir}/{textbook['textbook_pdf']}"
            a_path = q_path
        else:
            q_path = f"{textbook_dir}/{textbook['question_pdf']}"
            a_path = f"{textbook_dir}/{textbook['solutions_pdf']}"
        pdfs_to_process.append((q_path, a_path))

    Preprocessor(pdfs_to_process, impute=not args.no_impute)()


# ---------------------------------------------------------------------------
# End of file!
# ---------------------------------------------------------------------------
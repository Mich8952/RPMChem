# preprocessing/get_jsons_joint.py

"""
Extracts question+solution pairs from a joint PDF (textbook + solutions in
one file) using a vision-language model via LM Studio.
"""

import base64
import gc
import json
import logging
import re
import time
import traceback
from datetime import datetime

from openai import OpenAI
from pdf2image import convert_from_path
from pypdf import PdfReader
from PIL import Image

from prompts import build_joint_prompt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_PAGES_PER_QUESTION = 3
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
MODEL_NAME = "qwen/qwen3-vl-30b"
API_SLEEP_SECONDS = 3


# ---------------------------------------------------------------------------
# Logging + client
# ---------------------------------------------------------------------------

datestamp = (
    str(datetime.now())
    .replace(" ", "__")
    .replace(".", "__")
    .replace(":", "_")
)

#checkpoint_file = f'datasets/processed_real/checkpoint_{datestamp}.txt' # in case of crashes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            f"datasets/processed_real/extraction_log_{datestamp}.log"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key="lm-studio")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_json_response(raw_text: str) -> dict:
    """Strip optional markdown fences and parse JSON."""
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip())
    return json.loads(cleaned)


def encode_pil_image_from_path(image_path: str) -> str:
    """Base64-encode a PNG file from disk."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def process_pages(
    pdf_path: str,
    start_page: int,
    num_pages: int,
    final_page: int,
) -> list[str]:
    """Render PDF pages to temporary PNGs and return as base64 strings."""
    images = []
    for offset in range(num_pages):
        page_num = start_page + offset
        if page_num > final_page:
            break
        pages = convert_from_path(
            pdf_path, dpi=1000, first_page=page_num, last_page=page_num
        )
        image_path = f"output_image_page_{page_num}.png"
        pages[0].save(image_path, "PNG")
        images.append(encode_pil_image_from_path(image_path))
        del pages
        gc.collect()
    return images


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_joint_pdf(
    pdf_path: str,
    output_path: str,
    initial_page: int = INITIAL_PAGE,
    final_page: int = FINAL_PAGE,
) -> None:
    """
    Iterate over pages of a joint PDF and write question/solution pairs to
    a JSONL file at *output_path*.

    Args:
        pdf_path: Relative path to the source PDF from the repo root.
        output_path: Destination JSONL file path.
        initial_page: First page to process.
        final_page: Last page to process.
    """
    page_index = initial_page

    with open(output_path, "a", encoding="utf-8") as out_file:
        while page_index <= final_page:
            try:
                best_result = None
                pages_used = 1

                for num_pages_to_try in range(1, MAX_PAGES_PER_QUESTION + 1):
                    if page_index + num_pages_to_try - 1 > final_page:
                        break

                    time.sleep(API_SLEEP_SECONDS)
                    base64_images = process_pages(
                        pdf_path, page_index, num_pages_to_try, final_page
                    )

                    content = [
                        {"type": "text", "text": build_joint_prompt(num_pages_to_try)}
                    ]
                    for img in base64_images:
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img}"
                                },
                            }
                        )

                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": content}],
                        max_tokens=3000,
                    )

                    parsed = parse_json_response(
                        completion.choices[0].message.content
                    )
                    questions = parsed.get("questions", [])

                    if questions:
                        best_result = parsed
                        pages_used = num_pages_to_try
                        logger.info(
                            "Page %d: extracted %d questions using %d pages: %s",
                            page_index,
                            len(questions),
                            num_pages_to_try,
                            [q["question_num"] for q in questions],
                        )
                        break

                    logger.info(
                        "Page %d: no questions at %d pages — %s",
                        page_index,
                        num_pages_to_try,
                        parsed.get("summary", "no summary"),
                    )
                    del base64_images, content, completion, parsed
                    gc.collect()

                if best_result:
                    for question in best_result["questions"]:
                        question["pages_used"] = pages_used
                        out_file.write(json.dumps(question) + "\n")
                    page_index += pages_used
                else:
                    logger.warning(
                        "Page %d: no valid extraction after %d attempts",
                        page_index,
                        MAX_PAGES_PER_QUESTION,
                    )
                    page_index += 1

                if page_index % 10 == 0:
                    gc.collect()

            except Exception:
                logger.error("Page %d: unhandled exception", page_index)
                logger.debug(traceback.format_exc())
                page_index += 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Edit conf/preprocessing.yaml to configure PDF paths.
    PDF_PATH = (
        "datasets/raw/"
        "Thomas Engel and Philip Reid - Solution Manual for Physical Chemistry (0) - libgen.li.pdf"
    )
    OUTPUT_PATH = f"datasets/processed_real/full_dataset_{datestamp}.jsonl"

    extract_joint_pdf(PDF_PATH, OUTPUT_PATH)


# ---------------------------------------------------------------------------
# End of file!
# ---------------------------------------------------------------------------
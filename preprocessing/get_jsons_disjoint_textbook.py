# preprocessing/get_jsons_disjoint_textbook.py

"""
Extracts question/answer items from disjoint textbook and solutions PDFs
using a vision-language model via an LM Studio OpenAI-compatible endpoint.
"""


import base64
import gc
import io
import importlib
import json
import logging
import re
import time
import traceback
from datetime import datetime

from openai import OpenAI
from pdf2image import convert_from_path
from pypdf import PdfReader

from prompts import build_disjoint_question_prompt, build_disjoint_answer_prompt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_PAGES_PER_ITEM = 3  # max num of consecutive pages to look at 
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
MODEL_NAME = "qwen/qwen3-vl-30b"
API_SLEEP_SECONDS = 3


# Want to define keywords that are specific to this textbook. Page will only be processed if keywords are found on the page (this is a quick heuristic to save cost by not running the vision model on pages that clearly dont have questions on them). This is especially important for this textbook because the questions are at the end of each chapter and there are a lot of pages that are just content. We want to make sure we can skip those content pages as much as possible.

PDF1_SECTION_KEYWORDS = [
    "discussion questions",
    "exercises",
    "problems",
    "exercises and problems",
]


# ---------------------------------------------------------------------------
# Logging + client (module-level singletons)
# ---------------------------------------------------------------------------

datestamp = (
    str(datetime.now())
    .replace(" ", "__")
    .replace(".", "__")
    .replace(":", "_")
)

# checkpoint_file = f'datasets/processed_real/checkpoint_{datestamp}.txt' # in case of crashes

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


def encode_pil_image(image) -> str:
    """Base64-encode a PIL image as PNG."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace for keyword matching."""
    return re.sub(r"\s+", " ", text.lower()).strip()


# ---------------------------------------------------------------------------
# PDF utilities
# ---------------------------------------------------------------------------

class PdfExtractor:
    """Extracts and normalises plain text from a PDF page."""

    def __init__(self, pdf_path: str):
        self.reader = PdfReader(pdf_path)
        self.num_pages = len(self.reader.pages)

    def __call__(self, page_num: int) -> str:
        page_idx = page_num - 1
        if page_idx < 0 or page_idx >= self.num_pages:
            return ""
        raw = self.reader.pages[page_idx].extract_text() or ""
        return normalize_text(raw)

def process_pages(
    pdf_path: str,
    start_page: int,
    num_pages: int,
    last_page: int,
) -> list[str]:
    """Render PDF pages to base64-encoded PNG strings."""
    images = []
    for offset in range(num_pages):
        page_num = start_page + offset
        if page_num > last_page:
            break
        pages = convert_from_path(
            pdf_path, dpi=500, first_page=page_num, last_page=page_num
        )
        images.append(encode_pil_image(pages[0]))   # encode the first (and only) page returned as PNG
        del pages                                   # clean up memory immediately after processing each page to avoid OOM
        gc.collect()
    return images

def should_scan_pdf1_page(
    page_num: int,
    text_extractor: PdfExtractor | None,
) -> bool:
    """Return True if the page likely contains exercise/problem content."""
    if text_extractor is None:
        return True
    page_text = text_extractor(page_num)
    if not page_text:
        return True
    return any(keyword in page_text for keyword in PDF1_SECTION_KEYWORDS)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_items_from_pdf(
    pdf_path: str,
    output_path: str,
    prompt_builder,
    final_page: int | None = None,
    should_scan_page=None,
) -> None:
    """
    Iterate over pages of *pdf_path*, call the vision model, and write
    extracted items to *output_path* as a JSON file.

    Args:
        pdf_path: Path to the source PDF.
        output_path: Destination JSON file path.
        prompt_builder: Callable(num_pages) -> str.
        final_page: Last page to process. Defaults to the total page count.
        should_scan_page: Optional callable(page_num) -> bool gating the
            vision call with a cheap text-based pre-filter.
    """
    extractor = PdfExtractor(pdf_path)
    if final_page is None:
        final_page = extractor.num_pages

    records = []
    page_index = 1

    while page_index <= final_page:
        try:
            best_result = None
            pages_used = 1

            if should_scan_page is not None and not should_scan_page(page_index):
                logger.info("Page %d: skipped (no section keywords)", page_index)
                page_index += 1
                continue

            for num_pages_to_try in range(1, MAX_PAGES_PER_ITEM + 1):
                if page_index + num_pages_to_try - 1 > final_page:
                    break

                time.sleep(API_SLEEP_SECONDS)
                base64_images = process_pages(
                    pdf_path=pdf_path,
                    start_page=page_index,
                    num_pages=num_pages_to_try,
                    last_page=final_page,
                )

                content = [{"type": "text", "text": prompt_builder(num_pages_to_try)}]
                for img in base64_images:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img}"},
                        }
                    )

                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=3000,
                )

                parsed = parse_json_response(completion.choices[0].message.content)
                items = parsed.get("items", [])

                if items:
                    best_result = parsed
                    pages_used = num_pages_to_try
                    logger.info(
                        "Page %d: extracted %d items using %d pages",
                        page_index,
                        len(items),
                        num_pages_to_try,
                    )
                    break

                logger.info(
                    "Page %d: no items at %d pages — %s",
                    page_index,
                    num_pages_to_try,
                    parsed.get("summary", "no summary"),
                )
                del base64_images, content, completion, parsed
                gc.collect()

            if best_result:
                for item in best_result["items"]:
                    item["pages_used"] = pages_used
                    item["source_page"] = page_index
                    records.append(item)
                page_index += pages_used
            else:
                logger.warning(
                    "Page %d: no valid extraction after %d attempts",
                    page_index,
                    MAX_PAGES_PER_ITEM,
                )
                page_index += 1

            if page_index % 10 == 0:
                gc.collect()

        except Exception:
            logger.error("Page %d: unhandled exception", page_index)
            logger.debug(traceback.format_exc())
            page_index += 1

    payload = {
        "source_pdf": pdf_path,
        "generated_at": datestamp,
        "initial_page": 1,
        "final_page": final_page,
        "records": records,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)

    logger.info("Saved %d records to %s", len(records), output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Edit conf/preprocessing.yaml to change PDF paths — do not hardcode here.
    # These defaults match the paths defined in that config file.
    PDF1 = "datasets/pdfs/Atkins_ Physical Chemistry 11e.pdf"
    PDF2 = "datasets/pdfs/Solutions Manual - Atkins Physical Chemistry 11th Ed.pdf"

    PDF1_JSON = "datasets/processed_real/pdf1_json.json"
    PDF2_JSON = "datasets/processed_real/pdf2_json.json"

    pdf1_extractor = PdfExtractor(PDF1)

    extract_items_from_pdf(
        PDF1,
        PDF1_JSON,
        build_disjoint_question_prompt,
        final_page=pdf1_extractor.num_pages,
        should_scan_page=lambda p: should_scan_pdf1_page(p, pdf1_extractor),
    )
    extract_items_from_pdf(
        PDF2,
        PDF2_JSON,
        build_disjoint_answer_prompt,
    )


# ---------------------------------------------------------------------------
# End of file!
# ---------------------------------------------------------------------------
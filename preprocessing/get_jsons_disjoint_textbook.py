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
from pypdf import PdfReader

from openai import OpenAI
from pdf2image import convert_from_path




max_pages_per_item = 3  # max num of consecutive pages to look at 

datestamp = str(datetime.now()).replace(" ", "__").replace(".","__").replace(":","_") # timestamp for this run so that we can keep track

checkpoint_file = f'datasets/processed_real/checkpoint_{datestamp}.txt' # in case it crashes

logging.basicConfig( # logging config so I can see what the error is (I originally had a bug with LM Studio unloading my model halfway through the processoing)
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'datasets/processed_real/extraction_log_{datestamp}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")


# in this case, want to define keywords that are specific to this textbook. In this textbook, we want to look at pages that have any of these. If they have them, then we process. Otherwise, we would run the LLM on every single page and get it to form the same conclusion - takes forever.

pdf1_section_keywords = [
    "discussion questions",
    "exercises",
    "problems",
    "exercises and problems",
]


def string_parser(raw_text):
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip())
    return json.loads(cleaned)


def encode_pil_image(image): # this is an improvement over the last get_jsons (now called joint) because it doesnt operate on images. We can just run it directly from memory 
    buffer = io.BytesIO() # 
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def process_pages(pdf_path, start_page, num_pages, last_page):
    images = []
    for page_offset in range(num_pages):
        page_num = start_page + page_offset
        if page_num > last_page:
            break
        pages = convert_from_path(pdf_path, dpi=500, first_page=page_num, last_page=page_num)
        images.append(encode_pil_image(pages[0])) # base64 encode them so that we can pass them to language model
        del pages # then do memory cleanup as I was running OOM somehow before (accumulated ram each run somehow)
        gc.collect()
    return images

def normalize_text(text):
    return re.sub(r"\s+", " ", text.lower()).strip() # make it lower case and also remove whitespace. THe idea is that we want the quick tester (that looks for section keywords) to be able to actually find the keywords even if there are textual artifacts.


class PdfExtractor:
    def __init__(self,pdf_path):
        self.reader = PdfReader(pdf_path)
    def __call__(self,page_num):
        page_idx = page_num-1
        if page_idx < 0 or page_idx >= len(self.reader.pages):
            return ""
        
        raw = self.reader.pages[page_idx].extract_text() or ""

        return normalize_text(raw)


def should_scan_pdf1_page(page_num, text_extractor): # boolean function basically to say "should we actually scan this for questions" - which occurs if and only if it surpasses the first check
    if text_extractor is None:
        return True

    page_text = text_extractor(page_num)
    if not page_text:
        return True

    return any(keyword in page_text for keyword in pdf1_section_keywords)


def prompt_pdf1_questions(num_pages_to_try):
    return f"""
You are an advanced OCR machine.

You are shown {num_pages_to_try} consecutive page(s) from a question PDF.

Extract all COMPLETE questions visible on these pages and return JSON only.

Output schema:
{{
  "items": [
    {{
      "question_num": "string",
      "question_text": "string"
    }}
  ],
  "pages_used": {num_pages_to_try},
  "summary": "string"
}}

Rules:
- Include only questions where both number and full question text are fully visible.
- Exclude anything with missing/cut-off text.
- Use ASCII text only.
- Preserve equations and chemistry notation in text form.
- If none are complete, return an empty items list.
"""


def prompt_pdf2_answers(num_pages_to_try):
    return f"""
You are an advanced OCR machine.

You are shown {num_pages_to_try} consecutive page(s) from an answers/solutions PDF.

Extract all COMPLETE answer entries visible on these pages and return JSON only.

Output schema:
{{
  "items": [
    {{
      "question_num": "string",
      "answer_text": "string"
    }}
  ],
  "pages_used": {num_pages_to_try},
  "summary": "string"
}}

Rules:
- Include only entries where question number and full answer text are fully visible.
- Exclude anything with missing/cut-off text.
- Use ASCII text only.
- Preserve equations and chemistry notation in text form.
- If none are complete, return an empty items list.
"""


def extract_items_from_pdf(pdf_path, output_path, prompt_builder, should_scan_page=None):
    records = []
    initial_page = 1
    page_index = initial_page

    final_page = len(PdfExtractor(pdf_path).reader.pages) # kind of bad practice for now but will revisit later

    while page_index <= final_page:
        try:
            best_result = None
            pages_used = 1

            if should_scan_page is not None and not should_scan_page(page_index):
                logger.info(
                    f"Page {page_index}: skipped before vision model (missing section keywords)"
                )
                page_index += 1
                continue

            for num_pages_to_try in range(1, max_pages_per_item + 1):
                if page_index + num_pages_to_try - 1 > final_page:
                    break

                time.sleep(3)
                base64_images = process_pages(pdf_path=pdf_path, start_page=page_index, num_pages=num_pages_to_try, last_page=final_page)

                content = [{"type": "text", "text": prompt_builder(num_pages_to_try)}]
                for base64_img in base64_images:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_img}"},
                        }
                    )

                completion = client.chat.completions.create(
                    model="qwen/qwen3-vl-30b",
                    messages=[{"role": "user", "content": content}],
                    max_tokens=3000, # increase if we hit the limit but it likely wont happen given we set max pages to 3 
                )

                parsed = string_parser(completion.choices[0].message.content)
                items = parsed.get("items")

                if items != []:
                    best_result = parsed
                    pages_used = num_pages_to_try
                    logger.info(f"Page at {page_index}: extracted {len(items)} qs using {num_pages_to_try} pages")
                    break

                logger.info(f"Page {page_index}: no valid items at {num_pages_to_try} pages, {parsed.get('summary', 'no summary was found')}")

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
                    f"Page {page_index}: no valid extraction up to {max_pages_per_item} pages"
                )
                page_index += 1

            if page_index % 10 == 0:
                gc.collect()
                logger.info(f"Memory cleanup at page {page_index}")

        except Exception as error:
            logger.error(
                f"Page {page_index}: failed with {type(error).__name__}: {str(error)}"
            )
            logger.debug(f"Page {page_index}: traceback:\n{traceback.format_exc()}")
            page_index += 1

    payload = {
        "source_pdf": pdf_path,
        "generated_at": datestamp,
        "initial_page": initial_page,
        "final_page": final_page,
        "records": records,
    }

    
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=True, indent=2)

    logger.info(f"Saved {len(records)} records to {output_path}")


if __name__ == "__main__":


    pdf1 = "/Users/michaelmurray/Documents/GitHub/RPMChem/datasets/pdfs/Atkins_ Physical Chemistry 11e.pdf"
    pdf2 = "/Users/michaelmurray/Documents/GitHub/RPMChem/datasets/pdfs/Solutions Manual - Atkins Physical Chemistry 11th Ed.pdf"

    pdf1_json = "datasets/processed_real/pdf1_json.json"
    pdf2_json = "datasets/processed_real/pdf2_json.json"

    pdf1_text_extractor = PdfExtractor(pdf1)

    extract_items_from_pdf(
        pdf1,
        pdf1_json,
        prompt_pdf1_questions,
        should_scan_page=lambda page_num: should_scan_pdf1_page(page_num, pdf1_text_extractor),
    )
    extract_items_from_pdf(pdf2, pdf2_json, prompt_pdf2_answers)


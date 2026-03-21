"""
Script that processes the pdf file into a jsonl file with question and answer pairs. 
Some notes;
- We tried using original OCR, but it was really bad at the subscripts
- We found that this vision model was much better at extracting the text with high quality but it was slightly more computationally expensive
    - This is fine because I just run it locally overnight and it wasnt too bad


Code is not setup in OOP form because this is not re-used in other modules
"""

import lmstudio as lms
from tqdm import tqdm
from pydantic import BaseModel
from pdf2image import convert_from_path
from datetime import datetime
from PIL import Image
import logging
import traceback
import gc
import os
model = lms.llm('qwen/qwen3-vl-30b')

from openai import OpenAI
import re
import json
string_parser = lambda x: json.loads(re.sub(r"^```json\s*|\s*```$", "", x.strip()))

import base64
import time

client = OpenAI(api_key="lm-studio")

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

initial_page = 10
final_page = 755
max_pages_per_question = 3  # max num of consecutive pages to look at 

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


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8") # these models require us to base64 encode the images (the pages)

def process_pages(start_page, num_pages):
    images = []
    for page_offset in range(num_pages): # go through each page 
        page_num = start_page + page_offset
        if page_num > final_page:
            break
        
        pages = convert_from_path( 
            "/Users/michaelmurray/Documents/GitHub/RPMChem/datasets/raw/Thomas Engel and Philip Reid - Solution Manual for Physical Chemistry (0) - libgen.li.pdf", 
            1000,  # 1000 DPI is a little extreme but I didn't want to miss anything major
            first_page=page_num, 
            last_page=page_num
        )
        
        image_path = f"output_image_page_{page_num}.png" # save the page and image
        #pages[0].save(image_path, "PNG")
        images.append(encode_image(image_path))
        
        del pages
        gc.collect() # garbage collection because I was running into OOM issues for some reason.
        
    return images

res = []
i = initial_page

while i <= final_page: 
    try:
        best_result = None
        pages_used = 1
        
        for num_pages_to_try in range(1, max_pages_per_question + 1):
            if i + num_pages_to_try - 1 > final_page:
                break
                
            time.sleep(3)
            base64_images = process_pages(i, num_pages_to_try)
            
            content = [ # this docstring is ugly, but I dont know if adding tabs will show up in the actual string (I dont want) - best to call this from a separate file but for now this works
                {"type": "text", "text": f"""
You are an advanced OCR machine. Your task is to convert textbook questions and solutions into JSON format.

CRITICAL: You are being shown {num_pages_to_try} consecutive page(s). Extract ALL complete questions visible on these pages.
                         
CRITICAL TEXT FORMATTING RULES:
- Use plain ASCII characters only
- Convert superscripts to caret notation: write "mol⁻¹" as "mol^-1"
- Convert subscripts to underscore notation: write "H₂O" as "H_2O"  
- Use "x" for multiplication instead of ×: write "2.5 × 10" as "2.5 x 10"
- Use standard characters: ° as "degrees", ± as "+/-", etc.
- DO NOT use Unicode escape sequences like \\u00d7 or \\u207b

Example conversions:
- "8.314 J mol⁻¹ K⁻¹" → "8.314 J mol^-1 K^-1"
- "2.5 × 10³" → "2.5 x 10^3"
- "H₂SO₄" → "H_2SO_4"

CRITICAL VALIDATION RULES FOR EACH QUESTION:
1. You must be able to see BOTH the question number AND where the question ends clearly.
2. The ENTIRE solution must be visible with no text cut off at the bottom or continuing beyond the visible pages.
3. If any part of the solution is cut off (even one word), DO NOT include that question.
4. If the question or solution requires visual elements to understand or solve, DO NOT include it. This includes:
   - Graphs, plots, or charts
   - Molecular structure drawings (like benzene rings, Lewis structures, 3D structures)
   - Phase diagrams
   - Apparatus diagrams or experimental setups
   - Tables with data that cannot be easily transcribed as text
   
   IMPORTANT: Mathematical equations and chemical reaction equations are ALLOWED and should be transcribed.
   For example, "2H_2 + O_2 -> 2H_2O" or "PV = nRT" must be included.
   
5. The question and solution can be transcribed entirely as text without losing critical information.

EXTRACTION INSTRUCTIONS:
- Extract ALL questions where you can see the complete question AND complete solution
- For each question, identify where it starts and where it ends
- If a question's solution continues beyond the visible pages, DO NOT include it
- Return an array of all valid questions found

JSON Format - Return an array of questions:
{{
    "questions": [
        {{
            "valid": true,
            "question_num": "string",
            "prompt": "string", 
            "completion": "string",
            "next_question_num": "string"
        }},
        ... more questions ...
    ],
    "pages_used": {num_pages_to_try},
    "summary": "brief summary like 'Found X complete questions on Y pages'"
}}

Rules:
- Only include questions where BOTH the question AND the complete solution are visible
- For each question, fill in next_question_num with the number of the question that follows it
- Transcribe exactly what you see - no additions or assumptions
- Include the complete solution in "completion", not just the final answer
- If questions span multiple pages, combine the text appropriately
- If NO valid questions are found, return an empty questions array
                                """
                }
            ] # I used ChatGPT to help refine this prompt and it seems to work quite well.
            

            for base64_img in base64_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_img}"
                    }
                })

        

            completion = client.chat.completions.create(
                model="qwen/qwen3-vl-30b", # doesnt really matter tbh, I perfer to have control over which specific model to use (gotta use the 5bit mlx one for some reason)
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                max_tokens=3000, # annoying but some questions/solutions are very long so I had to increase this (this seems to be okay). Otherwise you get truncation which is bad.
            )

            jsoned_dict = string_parser(completion.choices[0].message.content)
            
            questions = jsoned_dict.get("questions", [])
            if questions and len(questions) > 0: # if we found any valid questions, then we can stop looking at more pages and save the result (if not then we tryy again with more pages in case the solution is cut off)
                best_result = jsoned_dict
                pages_used = num_pages_to_try
                logger.info(f"Page {i}: Successfully extracted {len(questions)} complete question using {num_pages_to_try} pages: {[q['question_num'] for q in questions]}")
                break
            else:
                logger.info(f"Page {i}: Trying with {num_pages_to_try} pages. Cant find valid questions: {jsoned_dict.get('summary', 'no summary')}")
            
            del base64_images, content, completion, jsoned_dict
            gc.collect() # clean up memory cuz oom issues (literally so annoying)
        
        if best_result: # save periodically
            with open(f'datasets/processed_real/full_dataset_{datestamp}.jsonl', "a", encoding="utf-8") as f:
                for question in best_result["questions"]:
                    question["pages_used"] = pages_used
                    f.write(json.dumps(question))
                    f.write("\n") # need this before line
            i += pages_used 
        else:
            logger.warning(f"Page {i}: Could not extract valid questions even after trying up to {max_pages_per_question} pages")
            i += 1  
        
        if i % 10 == 0:
            gc.collect() # idk if this is necessary again, but just wanted to be safe and it doesn't break anything
            logger.info(f"Memory cleanup performed at {i}")

    except Exception as e:
        logger.error(f"Page {i}: Failed with error: {type(e).__name__}: {str(e)}") # this is important because we want to know what cause the error (especially with try statements it becomes hard to debug)
        logger.debug(f"Page {i}: Full traceback:\n {traceback.format_exc()}")
        i += 1  

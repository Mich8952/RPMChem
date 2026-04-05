# preprocessing/prompts.py
"""LLM prompts for PDF extraction."""

# ---------------------------------------------------------------------------

def build_disjoint_question_prompt(num_pages: int) -> str:
    """Prompt for extracting questions from a standalone textbook PDF."""
    return f"""\
You are an advanced OCR machine.

You are shown {num_pages} consecutive page(s) from a question PDF.

Extract all COMPLETE questions visible on these pages and return JSON only.

Output schema:
{{
  "items": [
    {{
      "question_num": "string",
      "question_text": "string"
    }}
  ],
  "pages_used": {num_pages},
  "summary": "string"
}}

Rules:
- Include only questions where both number and full question text are fully visible.
- Exclude anything with missing/cut-off text.
- Use ASCII text only.
- Preserve equations and chemistry notation in text form.
- If none are complete, return an empty items list.
"""

# ---------------------------------------------------------------------------

def build_disjoint_answer_prompt(num_pages: int) -> str:
    """Prompt for extracting answers from a solutions manual PDF."""
    return f"""\
You are an advanced OCR machine.

You are shown {num_pages} consecutive page(s) from an answers/solutions PDF.

Extract all COMPLETE answer entries visible on these pages and return JSON only.

Output schema:
{{
  "items": [
    {{
      "question_num": "string",
      "answer_text": "string"
    }}
  ],
  "pages_used": {num_pages},
  "summary": "string"
}}

Rules:
- Include only entries where question number and full answer text are fully visible.
- Exclude anything with missing/cut-off text.
- Use ASCII text only.
- Preserve equations and chemistry notation in text form.
- If none are complete, return an empty items list.
"""

# ---------------------------------------------------------------------------

def build_joint_prompt(num_pages: int) -> str:
    """Prompt for extracting question+solution pairs from a joint PDF."""
    return f"""\
You are an advanced OCR machine. Your task is to convert textbook questions \
and solutions into JSON format.

CRITICAL: You are being shown {num_pages} consecutive page(s). Extract ALL complete questions visible on these pages.
                         
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
    "pages_used": {num_pages},
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

# ---------------------------------------------------------------------------

REASONING_SYSTEM_PROMPT = (
    "You are an expert chemistry tutor. Given a chemistry question and a "
    "ground-truth final solution, generate only the reasoning steps that lead "
    "to that solution. Be numerically and chemically consistent. "
    "Do not restate the final solution verbatim."
)

# ---------------------------------------------------------------------------

REASONING_USER_TEMPLATE = """\
Question:
{prompt}

Ground-truth final solution:
{completion}

Task:
Write only the reasoning steps that lead to the provided final solution.
Requirements:
1) Keep equations and unit logic explicit.
2) Keep it concise but complete enough to reproduce the solution.
3) Do NOT include a final-answer section.
4) Do NOT add JSON or markdown fences.
"""

# ---------------------------------------------------------------------------

NUMBER_EXTRACTION_PROMPT = """\
You are given a question and its corresponding answer.
Your task is to extract a SINGLE final numerical value and its unit \
(including SI prefix) if one exists.

Rules:
1. If the answer contains exactly one unambiguous final numerical value, output it.
2. Preserve unit text and SI prefix exactly (examples: MW, kJ, cm^2, m s^-1, %).
3. If the result is dimensionless, set unit to NA.
4. If there is no numerical value, more than one possible final value, a range, \
or ambiguity, set both value and unit to NA.
5. If the question/answer pair is not relevant to a single final answer \
(e.g. chemistry formula balancing), set both to NA.
6. Convert symbolic final forms to decimal when feasible.
7. Output STRICT JSON only:
{"value": "<number or NA>", "unit": "<unit or NA>"}

Do not include any explanation or extra text.\
"""


# ---------------------------------------------------------------------------
# End of file!
# ---------------------------------------------------------------------------
import os
import json
from pathlib import Path
from google import genai
from google.genai import types
from PIL import Image

OCR_PROMPT = """
You are an expert newspaper archivist. 
Your task is to analyze the provided newspaper page image and generate a structured JSON representation of its full content.

**Instructions:**
1. **Analyze the Layout:** Identify every distinct section on the page (articles, ads, mastheads, legal notices, comics, weather, etc.).
2. **Dynamic Structure:** Create a JSON structure that best fits the specific content of this page. You are free to define keys and nesting as needed to accurately represent the hierarchy of the page.
3. **Categorization Guidelines:**
   - Group standard news stories together (e.g., under "articles").
   - Group commercial content together (e.g., under "advertisements").
   - Identify meta-information (publication date, page number).
   - If unique sections appear (like "Radio Schedule" or "Comics"), create new top-level keys for them.
4. **Content Extraction:**
   - Transcribe headlines and body text exactly as written.
   - For images or visual ads, provide a visual description.

**Output Rule:**
Return only valid JSON. Do not include markdown formatting.
"""

def ocr_image_to_text(model: str, image_path: Path) -> str:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    img = Image.open(image_path)

    resp = client.models.generate_content(
        model=model,
        contents=[OCR_PROMPT, img],
        config=types.GenerateContentConfig(
            response_mime_type="text/plain",
        ),
    )
    if not resp.text:
        raise ValueError("Gemini response was empty; no text returned.")
    return resp.text

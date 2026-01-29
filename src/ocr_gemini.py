import os
import json
from pathlib import Path
from google import genai
from google.genai import types
from PIL import Image

OCR_PROMPT = """
You are an expert newspaper archivist and OCR transcriber.

Your task is to convert the provided newspaper page image into a structured JSON representation of the page AS A NEWSPAPER, organized by sections (not by arbitrary blocks).

NON-NEGOTIABLE RULES:
1) Do NOT guess or complete text. If something is unclear, use the token "[[ILLEGIBLE]]" for that span.
2) Do NOT repeat content. If you notice you are repeating the same phrase/lines, STOP and mark the remaining content of that section as unreadable.
3) Only include content that is visible in the image.
4) Return ONLY valid JSON (no markdown, no commentary).

GOAL:
Represent the page the way a reader experiences it: metadata → page sections → each section contains column-ordered items (headlines, notices, ads, listings, maps).

OUTPUT JSON FORMAT (strict):
{
  "metadata": {
    "publication": string|null,
    "date": string|null,
    "page_number": string|null,
    "edition_or_section_label": string|null
  },
  "sections": {
    "<SECTION_NAME>": {
      "section_label": string, 
      "items": [
        {
          "headline": string|null,
          "subheadline": string|null,
          "byline": string|null,
          "body": string|null
        }
      ],
      "unreadable_remainder": boolean,
      "unreadable_reason": string|null
    }
  },
  "reading_order": [ "<SECTION_NAME_1>", "<SECTION_NAME_2>", "..."]
}

SECTIONING INSTRUCTIONS:
- Create section names based on visible headers and natural newspaper groupings, for example:
  "Legal Notices", "Classified Ads", "Zoning Map", "Business Opportunities", "Help Wanted", etc.
- If a large region is clearly classifieds, you may split into subsections (e.g., "Classifieds - Help Wanted", "Classifieds - Real Estate") ONLY if those headings are visible.
- Put each item inside the section where it appears, and list items in top-to-bottom reading order within that section (and left-to-right within columns).
- For dense classifieds: each listing should be one item with a short "headline" (if present) and the listing text in "body".
- For maps/graphics: put readable labels in "body". If text is unclear, use "[[ILLEGIBLE]]".

TRANSCRIPTION RULES:
- Preserve capitalization and punctuation as written.
- Preserve line breaks in "body" using "\n" where it helps maintain fidelity.
- If a headline is readable but the body is not, set body to null and use "[[ILLEGIBLE]]" in place of unclear spans.

ANTI-LOOP SAFETY:
Only stop if you are repeating the exact same full line or phrase multiple times in a row.
Do NOT stop for legitimate repeated structures (e.g., multiple lost-and-found listings that share a format but differ in content).
If you begin repeating the exact same full line or phrase multiple times in a row:
- Set the current section’s "unreadable_remainder" to true,
- Fill "unreadable_reason" with "model began repeating",
- Stop adding more items to that section, and continue with other sections if possible.

Now produce the JSON for the provided page image.

"""


def ocr_image_to_text(
    model: str,
    image_path: Path,
    max_output_tokens: int = 10000,
    max_dim: int | None = 3200,
) -> dict:
    # Accept either GEMINI_API_KEY or GOOGLE_API_KEY (fallback)
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Missing API key: set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
        )

    client = genai.Client(api_key=api_key)

    img = Image.open(image_path)

    # Downscale very large images to avoid upload/processing timeouts
    if max_dim is not None and max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    try:
        resp = client.models.generate_content(
            model=model,
            contents=[OCR_PROMPT, img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
                max_output_tokens=max_output_tokens,  # adjust to your needs / model limits
            ),
        )
    except Exception as e:
        raise RuntimeError(
            "Gemini request failed (503/timeout or network issue). "
            "Check your API key, network, and try again."
        ) from e

    if not getattr(resp, "text", None):
        raise ValueError("Gemini response was empty; no text returned.")
    try:
        return json.loads(resp.text)
    except json.JSONDecodeError as exc:
        snippet = resp.text[:300].replace("\n", "\\n")
        raise ValueError(f"Gemini response was not valid JSON: {snippet}") from exc

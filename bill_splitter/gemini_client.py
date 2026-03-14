from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Optional, Tuple

from google import genai
from google.genai import types

from bill_splitter.models import ReceiptExtract, LineItem, LineType
from bill_splitter.prompts import ExtractedReceipt, RECEIPT_EXTRACTION_PROMPT


_ALLOWED_MIME = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/webp",
}


def _clean_amount_str(s: str) -> str:
    """
    Normalize typical receipt numeric formats into a Decimal-friendly string.
    Examples:
      "€ 1,99" -> "1.99"
      "- 0,50" -> "-0.50"
      "1 234,56" -> "1234.56"
    """
    if s is None:
        return "0"

    t = s.strip()

    # Remove currency symbols and letters (EUR, €, etc.)
    t = re.sub(r"[^\d,\.\-\s]", "", t)

    # Remove spaces
    t = t.replace(" ", "")

    # If it contains comma but no dot, treat comma as decimal separator
    if "," in t and "." not in t:
        t = t.replace(",", ".")
    else:
        # If it contains both, assume commas are thousand separators -> remove commas
        t = t.replace(",", "")

    # Fix cases like "--1.00" or "+1.00"
    t = re.sub(r"^\++", "", t)
    t = re.sub(r"^-{2,}", "-", t)

    # Empty fallback
    return t if t else "0"


def _to_decimal(s: Optional[str]) -> Optional[Decimal]:
    if s is None:
        return None
    cleaned = _clean_amount_str(s)
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def make_client(api_key: str) -> genai.Client:
    """
    Create a Gemini client using a user-provided API key.
    """
    if not api_key or not api_key.strip():
        raise ValueError("API key is empty.")
    return genai.Client(api_key=api_key.strip())


def test_api_key(api_key: str, model: str) -> Tuple[bool, str]:
    """
    Small sanity check you can wire to a 'Test Key' button in Gradio.
    """
    try:
        client = make_client(api_key)
        r = client.models.generate_content(model=model, contents="ping")
        _ = r.text  # ensure we got a response
        return True, "Key works."
    except Exception as e:
        return False, f"Key test failed: {type(e).__name__}: {e}"


def extract_receipt(
    *,
    api_key: str,
    model: str,
    file_bytes: bytes,
    mime_type: str,
    receipt_id: str,
) -> ReceiptExtract:
    """
    Extract receipt line items from a PDF/image using Gemini structured output.

    Returns your internal ReceiptExtract (with generated line item IDs).
    """
    if mime_type not in _ALLOWED_MIME:
        raise ValueError(f"Unsupported mime_type: {mime_type}. Allowed: {sorted(_ALLOWED_MIME)}")

    client = make_client(api_key)

    part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)

    response = client.models.generate_content(
        model=model,
        contents=[part, RECEIPT_EXTRACTION_PROMPT],
        config={
            "response_mime_type": "application/json",
            "response_schema": ExtractedReceipt,
            "temperature": 0.0,
        },
    )

    # With response_schema + application/json, response.text is JSON.
    raw_text = response.text or ""
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        # Fallback: sometimes response.text can be already-dumped or contain extra quotes
        # Try one more normalization pass.
        try:
            data = json.loads(raw_text.strip().strip("`"))
        except Exception as e:
            raise ValueError(f"Failed to parse Gemini JSON response: {e}. Raw: {raw_text[:500]}")

    extracted = ExtractedReceipt.model_validate(data)

    # Convert to internal ReceiptExtract + LineItem with generated IDs
    internal_items = []
    for idx, li in enumerate(extracted.line_items, start=1):
        dec = _to_decimal(li.amount)
        if dec is None:
            # Skip or set warning; we keep it but set to 0 to avoid crashing.
            extracted.warnings.append(f"Could not parse amount '{li.amount}' for line '{li.description}'. Set to 0.")
            dec = Decimal("0")

        ltype = LineType(li.type.value)
        internal_items.append(
            LineItem(
                id=f"li_{idx}",
                description=li.description.strip(),
                amount=dec,
                type=ltype,
            )
        )

    detected_total = _to_decimal(extracted.detected_total)

    return ReceiptExtract(
        receipt_id=receipt_id,
        merchant=extracted.merchant,
        date=extracted.date,
        currency=extracted.currency,
        line_items=internal_items,
        detected_total=detected_total,
        warnings=list(extracted.warnings or []),
    )

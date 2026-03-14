from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class ExtractedLineType(str, Enum):
    item = "item"
    discount = "discount"
    other = "other"


class ExtractedLineItem(BaseModel):
    """
    LLM-friendly line item schema.
    Amount is a string so the model doesn't output weird floats; we normalize to Decimal later.
    """
    description: str = Field(..., min_length=1, description="Receipt line description")
    amount: str = Field(..., description="Signed number as string, dot decimal. Example: '3.49' or '-1.00'")
    type: ExtractedLineType = Field(..., description="item | discount | other")


class ExtractedReceipt(BaseModel):
    merchant: Optional[str] = Field(None, description="Store name if found")
    date: Optional[str] = Field(None, description="Date string as shown on receipt")
    currency: Optional[str] = Field(None, description="Currency code if detectable, e.g. 'EUR'")
    detected_total: Optional[str] = Field(None, description="Total payable amount if found, signed string number")
    line_items: List[ExtractedLineItem] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


RECEIPT_EXTRACTION_PROMPT = """
You are extracting a supermarket receipt into structured JSON.

Rules:
- Output MUST conform to the provided JSON schema.
- Create a line for each purchase item (positive amount).
- Discounts MUST be returned as separate line items with NEGATIVE amounts and type="discount".
- Ignore payment method lines, VAT breakdown, change, and "TOTAL/SUBTOTAL" lines as split items.
  Those can be returned as type="other" only if you are unsure, but prefer to omit them from line_items.
- If you can find the final total payable, set detected_total.
- Amount format: use dot decimals (e.g. 1.99), no currency symbols, no thousands separators.
- If quantities are included, you may keep them inside description (MVP).
- Do not invent items. If unreadable, add a warning.

Extract the receipt now.
""".strip()

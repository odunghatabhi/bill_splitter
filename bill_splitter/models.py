from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional
from decimal import Decimal
from pydantic import BaseModel, Field


class LineType(str, Enum):
    item = "item"
    discount = "discount"
    other = "other"  # totals, VAT lines, payment method, change, etc.


class LineItem(BaseModel):
    """
    One row from the receipt that can be split across people.
    Amount is signed:
      - items usually positive
      - discounts negative
    """
    id: str = Field(..., description="Stable ID for this line item")
    description: str = Field(..., min_length=1)
    amount: Decimal = Field(..., description="Signed amount for this line item")
    type: LineType = Field(default=LineType.item)


class ReceiptExtract(BaseModel):
    """
    Output of the extraction step (Gemini -> structured data).
    line_items should include items + discounts.
    other_lines can include totals/tax/etc if you want to show them, but they won't be split.
    """
    receipt_id: str = Field(..., description="Stable ID for this receipt/session")
    merchant: Optional[str] = None
    date: Optional[str] = None  # keep as string for MVP; parse later if needed
    currency: Optional[str] = None  # e.g., EUR

    line_items: List[LineItem] = Field(default_factory=list)
    detected_total: Optional[Decimal] = None  # total payable if found

    warnings: List[str] = Field(default_factory=list)


class ComputeRequest(BaseModel):
    """
    Input to the splitting engine.
    allocations is sparse: missing/absent person means 0 share for that person.
    allocations[line_item_id][person_name] = weight (Decimal), e.g.:
      {"li_1": {"Alice": 1, "Bob": 2}, "li_2": {"Alice": 1}}
    """
    receipt_id: str
    people: List[str] = Field(default_factory=list)
    line_items: List[LineItem] = Field(default_factory=list)

    allocations: Dict[str, Dict[str, Decimal]] = Field(default_factory=dict)

    rounding_decimals: int = 2  # you said 2 decimals


class LineShare(BaseModel):
    """
    Optional detailed per-line breakdown (useful for UI).
    """
    line_item_id: str
    per_person: Dict[str, Decimal] = Field(default_factory=dict)
    line_total: Decimal = Decimal("0")


class ComputeResult(BaseModel):
    """
    Output of the splitting engine.
    totals: per-person totals (2 decimals).
    allocated_sum: sum of all per-person totals (should match sum of split line items).
    rounding_notes: record where tiny cent adjustments happened.
    """
    receipt_id: str
    totals: Dict[str, Decimal] = Field(default_factory=dict)

    line_shares: List[LineShare] = Field(default_factory=list)

    allocated_sum: Decimal = Decimal("0")
    receipt_sum: Decimal = Decimal("0")  # sum of item+discount line_items

    rounding_notes: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

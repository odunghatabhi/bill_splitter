from __future__ import annotations

from decimal import Decimal
from typing import Optional, Tuple, List

from bill_splitter.models import ReceiptExtract, LineType


def _sum_splitable_lines(receipt: ReceiptExtract) -> Decimal:
    """
    Sum only split-able lines: items + discounts.
    Ignore LineType.other.
    """
    total = Decimal("0")
    for li in receipt.line_items:
        if li.type in (LineType.item, LineType.discount):
            total += li.amount
    return total


def validate_extraction(
    receipt: ReceiptExtract,
    tolerance: Decimal = Decimal("0.02"),
) -> ReceiptExtract:
    """
    Adds warnings onto ReceiptExtract:
    - compares sum(line_items) to detected_total when available
    - checks for empty extraction
    - checks for suspicious cases (all 'other', etc.)
    """
    warnings: List[str] = list(receipt.warnings or [])

    if not receipt.line_items:
        warnings.append("No line items extracted. Please upload a clearer image/PDF or edit manually.")
        receipt.warnings = warnings
        return receipt

    split_sum = _sum_splitable_lines(receipt)

    # Basic sanity: if all lines are 'other', user can't split anything
    if all(li.type == LineType.other for li in receipt.line_items):
        warnings.append("All extracted lines are marked as 'other' (nothing to split). Please review extraction rules.")

    # Total check (if Gemini found a total)
    if receipt.detected_total is not None:
        diff = receipt.detected_total - split_sum
        if diff.copy_abs() > tolerance:
            warnings.append(
                f"Mismatch: detected total ({receipt.detected_total}) vs sum of items+discounts ({split_sum}) "
                f"diff = {diff}. Please review extracted lines."
            )

    # Useful hint if there are 'other' lines (totals/VAT/payment) and user might wonder
    if any(li.type == LineType.other for li in receipt.line_items):
        warnings.append("Note: 'other' lines (totals/VAT/payment) are not included in splitting.")

    receipt.warnings = warnings
    return receipt


def extraction_summary(receipt: ReceiptExtract) -> Tuple[Decimal, Optional[Decimal], List[str]]:
    """
    Convenience helper for UI: returns (split_sum, detected_total, warnings).
    """
    split_sum = _sum_splitable_lines(receipt)
    return split_sum, receipt.detected_total, list(receipt.warnings or [])

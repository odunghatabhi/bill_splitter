from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN
from typing import Dict, List, Tuple
import hashlib
import random

from bill_splitter.models import ComputeRequest, ComputeResult, LineShare, LineItem, LineType


CENT = Decimal("0.01")


def _to_cents(amount: Decimal, rounding_decimals: int = 2) -> int:
    """
    Convert a Decimal amount to integer cents, rounding HALF_UP to the requested decimals.
    Assumes currency-like values.
    """
    q = Decimal("1").scaleb(-rounding_decimals)  # e.g. 0.01 for 2 decimals
    rounded = amount.quantize(q, rounding=ROUND_HALF_UP)
    factor = 10 ** rounding_decimals
    cents = int((rounded * factor).to_integral_value(rounding=ROUND_HALF_UP))
    return cents


def _from_cents(cents: int, rounding_decimals: int = 2) -> Decimal:
    factor = Decimal(10) ** rounding_decimals
    return (Decimal(cents) / factor).quantize(Decimal("1").scaleb(-rounding_decimals))


def _stable_seed_int(receipt_id: str, line_item_id: str) -> int:
    """
    Deterministic seed from receipt + line id.
    Produces stable pseudo-randomness across runs.
    """
    s = f"{receipt_id}::{line_item_id}".encode("utf-8")
    h = hashlib.sha256(s).hexdigest()
    return int(h[:16], 16)  # enough bits for stable PRNG seed


def _stable_random_order(receipt_id: str, line_item_id: str, people: List[str]) -> List[str]:
    people = list(people)
    rng = random.Random(_stable_seed_int(receipt_id, line_item_id))
    rng.shuffle(people)
    return people


def _split_line_amount_cents(
    receipt_id: str,
    line_item_id: str,
    amount_cents: int,
    weights: Dict[str, Decimal],
) -> Tuple[Dict[str, int], List[str]]:
    """
    Split an integer amount of cents across people proportionally to weights.
    Returns:
      - per_person_cents
      - rounding_notes for that line
    """
    notes: List[str] = []

    # Keep only strictly positive weights; blank/missing/zero means 0 share
    cleaned = {p: w for p, w in weights.items() if w is not None and w > 0}

    if not cleaned:
        # No allocations: nobody shares this line
        return {}, [f"Line {line_item_id}: no allocations (unassigned amount {amount_cents} cents)."]

    people = list(cleaned.keys())
    total_w = sum(cleaned.values(), Decimal("0"))
    if total_w <= 0:
        return {}, [f"Line {line_item_id}: invalid total weight (unassigned amount {amount_cents} cents)."]

    # Proportional base allocation with ROUND_DOWN (toward zero for both signs)
    per_person: Dict[str, int] = {}
    sum_base = 0
    for p in people:
        exact = (Decimal(amount_cents) * cleaned[p]) / total_w
        base = int(exact.to_integral_value(rounding=ROUND_DOWN))
        per_person[p] = base
        sum_base += base

    remainder = amount_cents - sum_base

    if remainder != 0:
        # Distribute +/- 1 cent remainders in a stable-random order
        order = _stable_random_order(receipt_id, line_item_id, people)

        step = 1 if remainder > 0 else -1
        remaining = abs(remainder)
        idx = 0
        while remaining > 0:
            p = order[idx % len(order)]
            per_person[p] += step
            remaining -= 1
            idx += 1

        notes.append(
            f"Line {line_item_id}: distributed remainder {remainder} cent(s) "
            f"using stable-random order {order}."
        )

    return per_person, notes


def compute_splits(req: ComputeRequest) -> ComputeResult:
    """
    Main engine:
    - Computes per-person totals based on allocations/weights per line item.
    - Rounds to req.rounding_decimals (default 2).
    - Uses stable-random remainder assignment per line.
    """
    rounding_decimals = int(req.rounding_decimals or 2)

    # Initialize totals for provided people (preserve order for UI friendliness)
    totals_cents: Dict[str, int] = {p: 0 for p in req.people}
    line_shares: List[LineShare] = []
    rounding_notes: List[str] = []
    warnings: List[str] = []

    # Receipt sum should reflect split-able lines (items + discounts)
    receipt_sum_cents = 0
    allocated_sum_cents = 0

    for li in req.line_items:
        if li.type == LineType.other:
            continue  # ignore totals, VAT breakdown, etc.

        amount_cents = _to_cents(li.amount, rounding_decimals)
        receipt_sum_cents += amount_cents

        weights = req.allocations.get(li.id, {}) or {}

        per_person_cents, notes = _split_line_amount_cents(
            receipt_id=req.receipt_id,
            line_item_id=li.id,
            amount_cents=amount_cents,
            weights=weights,
        )
        rounding_notes.extend(notes)

        if not per_person_cents:
            # Unassigned line (no allocations)
            warnings.append(f"Line {li.id} unassigned: {li.description} ({_from_cents(amount_cents, rounding_decimals)}).")
            # Still include a line share record so the UI can show it's unallocated if desired
            line_shares.append(
                LineShare(
                    line_item_id=li.id,
                    per_person={},
                    line_total=_from_cents(0, rounding_decimals),
                )
            )
            continue

        # Update totals; include people not in req.people if user typed a new name mid-flow
        for p, cents in per_person_cents.items():
            if p not in totals_cents:
                totals_cents[p] = 0
            totals_cents[p] += cents
            allocated_sum_cents += cents

        # Store per-line breakdown as Decimal amounts (2 decimals)
        line_shares.append(
            LineShare(
                line_item_id=li.id,
                per_person={p: _from_cents(c, rounding_decimals) for p, c in per_person_cents.items()},
                line_total=_from_cents(sum(per_person_cents.values()), rounding_decimals),
            )
        )

    result = ComputeResult(
        receipt_id=req.receipt_id,
        totals={p: _from_cents(c, rounding_decimals) for p, c in totals_cents.items()},
        line_shares=line_shares,
        allocated_sum=_from_cents(allocated_sum_cents, rounding_decimals),
        receipt_sum=_from_cents(receipt_sum_cents, rounding_decimals),
        rounding_notes=rounding_notes,
        warnings=warnings,
    )

    # Extra warning if not everything got assigned
    if allocated_sum_cents != receipt_sum_cents:
        diff = receipt_sum_cents - allocated_sum_cents
        result.warnings.append(
            f"Allocated sum differs from receipt sum by {_from_cents(diff, rounding_decimals)} "
            f"(some lines may be unassigned)."
        )

    return result

from decimal import Decimal
from bill_splitter.models import ComputeRequest, LineItem, LineType
from bill_splitter.splitter import compute_splits

req = ComputeRequest(
    receipt_id="r1",
    people=["Alice", "Bob", "Cara"],
    line_items=[
        LineItem(id="l1", description="Milk", amount=Decimal("2.99"), type=LineType.item),
        LineItem(id="l2", description="Bread", amount=Decimal("1.50"), type=LineType.item),
        LineItem(id="l3", description="Discount coupon", amount=Decimal("-0.49"), type=LineType.discount),
    ],
    allocations={
        "l1": {"Alice": Decimal("1"), "Bob": Decimal("1")},        # split 50/50
        "l2": {"Bob": Decimal("1")},                               # Bob pays all
        "l3": {"Alice": Decimal("1"), "Bob": Decimal("1"), "Cara": Decimal("1")},  # split discount
    },
)

res = compute_splits(req)
print(res.model_dump())

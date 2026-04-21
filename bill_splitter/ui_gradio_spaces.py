from __future__ import annotations

from decimal import Decimal, InvalidOperation
import uuid
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from splitwise import Splitwise
from splitwise.expense import Expense
from splitwise.user import ExpenseUser

from bill_splitter.config import settings
from bill_splitter.gemini_client import extract_receipt, test_api_key
from bill_splitter.models import ComputeRequest
from bill_splitter.splitter import compute_splits
from bill_splitter import splitwise_app
from bill_splitter.validate import validate_extraction


DEFAULT_MODEL = settings.default_model
ALLOCATION_WEIGHT_START_COL = 3


# -------- helpers --------

def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _normalize_line_items_df(df: Any) -> List[Dict[str, Any]]:
    """
    Convert the editable gr.Dataframe into list of dicts.
    Expected columns: id, description, amount, type
    """
    if df is None:
        return []

    if isinstance(df, dict) and "data" in df:
        rows = df["data"]
        headers = df.get("headers") or []
    elif isinstance(df, list):
        rows = df
        headers = ["id", "description", "amount", "type"]
    else:
        try:
            headers = list(df.columns)
            rows = df.values.tolist()
        except Exception:
            return []

    items: List[Dict[str, Any]] = []
    for r in rows:
        if not r or all((c is None or str(c).strip() == "") for c in r):
            continue

        row = {}
        for i, h in enumerate(headers):
            row[h] = r[i] if i < len(r) else None

        li_id = _safe_str(row.get("id")).strip()
        desc = _safe_str(row.get("description")).strip()
        amt = _safe_str(row.get("amount")).strip()
        typ = _safe_str(row.get("type")).strip().lower() or "item"

        if not li_id:
            continue
        if not desc:
            desc = "(no description)"
        if not amt:
            amt = "0"

        items.append(
            {
                "id": li_id,
                "description": desc,
                "amount": amt,   # backend model parses Decimal
                "type": typ,
            }
        )
    return items


def _build_allocation_matrix(line_items: List[Dict[str, Any]], people: List[str]):
    """
    Allocation grid visible to user:
      headers: ["item", "price", "equal"] + people
      rows: for each line_item, [description, amount, False] + [""]*len(people)
    Also returns a hidden list of line_item_ids in the same row order.
    """
    people = [p.strip() for p in people if p and p.strip()]
    headers = ["item", "price", "equal"] + people
    datatypes = ["str", "str", "bool"] + ["str"] * len(people)
    rows: List[List[Any]] = []
    line_ids: List[str] = []

    for li in line_items:
        line_ids.append(li["id"])
        rows.append([li["description"], li.get("amount", ""), False] + [""] * len(people))

    return rows, headers, datatypes, line_ids


def _table_rows(df: Any) -> List[List[Any]]:
    if df is None:
        return []
    if isinstance(df, dict) and "data" in df:
        return df["data"] or []
    if isinstance(df, list):
        return df
    try:
        return df.values.tolist()
    except Exception:
        return []


def _is_checked(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _safe_str(value).strip().lower() in {"true", "1", "yes", "y", "checked", "on"}


def do_apply_equal_shares(alloc_df: Any, people_csv_val: str):
    people = [p.strip() for p in (people_csv_val or "").split(",") if p.strip()]
    rows = _table_rows(alloc_df)
    if not rows or not people:
        return gr.update(value=rows)

    changed = False
    target_len = ALLOCATION_WEIGHT_START_COL + len(people)
    normalized_rows: List[List[Any]] = []

    for raw_row in rows:
        row = list(raw_row or [])
        if len(row) < target_len:
            row.extend([""] * (target_len - len(row)))
        if len(row) > 2 and _is_checked(row[2]):
            for col_idx in range(ALLOCATION_WEIGHT_START_COL, target_len):
                if row[col_idx] != "1":
                    row[col_idx] = "1"
                    changed = True
        normalized_rows.append(row)

    if not changed:
        return gr.update(value=rows)
    return gr.update(value=normalized_rows)


def _allocations_df_to_dict(alloc_df: Any, people: List[str], line_ids: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Convert allocation matrix into sparse dict:
      { line_item_id: { person: weight_str } }
    The first visible column is 'item' (description), so we use `line_ids[row_index]`
    as the real line_item_id.
    Blank/invalid treated as absent (0 share).
    """
    if alloc_df is None:
        return {}

    people = [p.strip() for p in people if p and p.strip()]
    if not people or not line_ids:
        return {}

    rows = _table_rows(alloc_df)

    allocations: Dict[str, Dict[str, str]] = {}

    for row_idx, r in enumerate(rows):
        if row_idx >= len(line_ids):
            break
        line_id = str(line_ids[row_idx]).strip()
        if not line_id:
            continue

        per: Dict[str, str] = {}
        # New allocation tables use item/price/equal metadata columns before weights.
        # Keep old one-column tables readable for existing sessions.
        has_new_metadata_cols = len(r) >= ALLOCATION_WEIGHT_START_COL and (
            isinstance(r[2], bool) or _safe_str(r[2]).strip().lower() in {"true", "false", ""}
        )
        weight_start_col = ALLOCATION_WEIGHT_START_COL if has_new_metadata_cols else 1
        for i, person in enumerate(people, start=weight_start_col):
            if i >= len(r):
                continue
            s = _safe_str(r[i]).strip()
            if not s:
                continue
            per[person] = s

        if per:
            allocations[line_id] = per

    return allocations


def _guess_mime_from_name(name: str) -> str:
    name = (name or "").lower()
    if name.endswith(".pdf"):
        return "application/pdf"
    if name.endswith(".jpg") or name.endswith(".jpeg"):
        return "image/jpeg"
    if name.endswith(".png"):
        return "image/png"
    if name.endswith(".webp"):
        return "image/webp"
    return "application/octet-stream"


# -------- core actions (no HTTP) --------

def do_test_key(api_key: str, model: str) -> str:
    if not api_key or not api_key.strip():
        return "Enter an API key first."
    ok, msg = test_api_key(api_key=api_key, model=model)
    return f"{'✅' if ok else '❌'} {msg}"


def do_save_key(api_key_val: str,  model: str) -> Tuple[str, str, gr.update]:
    if not api_key_val or not api_key_val.strip():
        return "", "❌ Please enter an API key.", gr.update(visible=False)
    ok, msg = test_api_key(api_key=api_key_val, model=model)
    if ok:
        return api_key_val.strip(), "✅ Key saved for this session.", gr.update(visible=True)
    else:
        return "", "❌ Wrong API Key !!", gr.update(visible=False)


def do_extract(
    api_key: str,
    model: str,
    file_obj: Any,
) -> Tuple[str, str, List[List[Any]], List[str]]:
    """
    Returns:
      status_text, receipt_id, line_items_rows, alloc_line_ids(reset)
    """
    if not api_key or not api_key.strip():
        return "❌ No API key saved.", "", [], []

    if file_obj is None:
        return "❌ Please upload a file.", "", [], []

    file_path = getattr(file_obj, "name", None) or str(file_obj)
    mime_type = _guess_mime_from_name(file_path)

    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        rid = f"r_{uuid.uuid4().hex[:12]}"

        receipt = extract_receipt(
            api_key=api_key,
            model=model,
            file_bytes=file_bytes,
            mime_type=mime_type,
            receipt_id=rid,
        )

        receipt = validate_extraction(receipt)

        warnings = receipt.warnings or []
        wtxt = ""
        if warnings:
            wtxt = "Warnings:\n- " + "\n- ".join(str(x) for x in warnings)

        rows = []
        for li in receipt.line_items:
            rows.append([li.id, li.description, str(li.amount), li.type.value])

        status = "✅ Extracted line items."
        if wtxt:
            status += "\n\n" + wtxt

        # Reset allocation id mapping when a new receipt is extracted
        return status, receipt.receipt_id, rows, []

    except Exception as e:
        return f"❌ Extraction failed: {type(e).__name__}: {e}", "", [], []


def do_build_alloc(people_csv_val: str, line_items_df_val: Any):
    people = [p.strip() for p in (people_csv_val or "").split(",") if p.strip()]
    items = _normalize_line_items_df(line_items_df_val)
    rows, headers, datatypes, line_ids = _build_allocation_matrix(items, people)
    return gr.update(headers=headers, datatype=datatypes, value=rows), line_ids


def do_compute(
    receipt_id: str,
    people_csv: str,
    line_items_df: Any,
    alloc_df: Any,
    line_ids: List[str],
) -> str:
    if not receipt_id:
        return "❌ No receipt_id. Extract a receipt first."

    people = [p.strip() for p in (people_csv or "").split(",") if p.strip()]
    line_items = _normalize_line_items_df(line_items_df)
    allocations = _allocations_df_to_dict(alloc_df, people, line_ids or [])

    if not people:
        return "❌ Add at least one person (comma-separated)."
    if not line_items:
        return "❌ No line items. Extract or enter items first."

    req = ComputeRequest(
        receipt_id=receipt_id,
        people=people,
        line_items=line_items,     # Pydantic will coerce into LineItem list
        allocations=allocations,   # sparse dict
        rounding_decimals=2,
    )

    try:
        res = compute_splits(req)
        totals = res.totals or {}
        warnings = res.warnings or []

        lines: List[str] = []
        lines.append("✅ Totals per person:")
        for p in people:
            lines.append(f"- {p}: {totals.get(p, '0.00')}")
        extra_people = [k for k in totals.keys() if k not in people]
        for p in extra_people:
            lines.append(f"- {p}: {totals.get(p, '0.00')}")

        lines.append("")
        lines.append(f"Receipt sum (items+discounts): {res.receipt_sum}")
        lines.append(f"Allocated sum: {res.allocated_sum}")

        if warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in warnings:
                lines.append(f"- {w}")

        return "\n".join(lines)

    except Exception as e:
        return f"❌ Compute failed: {type(e).__name__}: {e}"


def _extract_splitwise_rows_from_compute_output(text: str) -> List[List[str]]:
    rows: List[List[str]] = []
    if not text:
        return rows

    in_totals = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.lower().startswith("✅ totals per person".lower()) or line.lower().startswith("âœ… totals per person".lower()):
            in_totals = True
            continue
        if in_totals and line == "":
            break
        if not in_totals:
            continue
        if not line.startswith("- ") or ":" not in line:
            continue
        person, amount = line[2:].split(":", 1)
        person = person.strip()
        amount = amount.strip()
        if person:
            rows.append([person, amount])
    return rows


def do_compute_with_splitwise(
    receipt_id: str,
    people_csv: str,
    line_items_df: Any,
    alloc_df: Any,
    line_ids: List[str],
) -> Tuple[str, List[List[str]]]:
    out = do_compute(receipt_id, people_csv, line_items_df, alloc_df, line_ids)
    return out, _extract_splitwise_rows_from_compute_output(out)


def _get_splitwise_client(api_key: str) -> Splitwise:
    return Splitwise(
        splitwise_app.CONSUMER_KEY,
        splitwise_app.CONSUMER_SECRET,
        api_key=api_key,
    )


def _member_display_name(member: Any) -> str:
    first = (member.getFirstName() or "").strip()
    last = (member.getLastName() or "").strip()
    return f"{first} {last}".strip() or f"User {member.getId()}"


def do_open_splitwise(splitwise_rows: List[List[str]]) -> Tuple[gr.update, str, gr.update, gr.update]:
    if not splitwise_rows:
        return (
            gr.update(visible=False),
            "Compute totals first, then open Splitwise.",
            gr.update(value=[]),
            gr.update(choices=[], value=None),
        )
    mapping_rows = [[r[0], r[1], ""] for r in splitwise_rows]
    people = [str(r[0]).strip() for r in mapping_rows if r and str(r[0]).strip()]
    return (
        gr.update(visible=True),
        "Splitwise panel opened. Connect and map each person to a group member.",
        gr.update(value=mapping_rows),
        gr.update(choices=people, value=(people[0] if people else None)),
    )


def do_splitwise_connect(api_key: str):
    if not api_key or not api_key.strip():
        return "Enter Splitwise API key.", "", gr.update(choices=[], value=None), {}, 0
    api_key = api_key.strip()
    try:
        s = _get_splitwise_client(api_key)
        current = s.getCurrentUser()
        current_id = int(current.getId()) if current else 0
        groups = s.getGroups()
    except Exception as e:
        return f"Splitwise connection failed: {type(e).__name__}: {e}", "", gr.update(choices=[], value=None), {}, 0

    group_map: Dict[str, int] = {}
    for g in groups or []:
        label = f"{g.getName()} ({g.getId()})"
        group_map[label] = int(g.getId())

    choices = list(group_map.keys())
    default_choice = choices[0] if choices else None
    if not choices:
        return "Connected. No groups found.", api_key, gr.update(choices=[], value=None), {}, current_id
    return "Connected. Select a group and load members.", api_key, gr.update(choices=choices, value=default_choice), group_map, current_id


def _load_splitwise_group_members(
    api_key: str,
    selected_group: str,
    group_map: Dict[str, int],
    mapping_source_rows: List[List[Any]],
    current_user_id: int,
):
    if not api_key:
        return (
            "Connect Splitwise first.",
            gr.update(value=[]),
            [],
            gr.update(choices=[], value=None),
            gr.update(value=[]),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
        )
    if not selected_group or selected_group not in group_map:
        return (
            "Select a valid group.",
            gr.update(value=[]),
            [],
            gr.update(choices=[], value=None),
            gr.update(value=[]),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
        )

    try:
        s = _get_splitwise_client(api_key)
        group = s.getGroup(int(group_map[selected_group]))
        members = group.getMembers() if hasattr(group, "getMembers") else []
        if not members and hasattr(group, "getUsers"):
            members = group.getUsers()
    except Exception as e:
        return (
            f"Failed to load members: {type(e).__name__}: {e}",
            gr.update(value=[]),
            [],
            gr.update(choices=[], value=None),
            gr.update(value=[]),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
        )

    member_rows: List[List[str]] = []
    member_state: List[Dict[str, Any]] = []
    member_names: List[str] = []
    for m in members or []:
        name = _member_display_name(m)
        uid = int(m.getId())
        member_rows.append([name, str(uid)])
        member_state.append({"id": uid, "name": name})
        member_names.append(name)

    mapping_rows: List[List[str]] = []
    for row in mapping_source_rows or []:
        person = str(row[0]).strip() if len(row) > 0 else ""
        amount = str(row[1]).strip() if len(row) > 1 else "0"
        existing_mapped = str(row[2]).strip() if len(row) > 2 else ""
        mapped = existing_mapped if existing_mapped in member_names else ""
        if not mapped and person in member_names:
            mapped = person
        mapping_rows.append([person, amount, mapped])

    payer_default = member_names[0] if member_names else None
    for m in member_state:
        if m["id"] == current_user_id:
            payer_default = m["name"]
            break

    people_choices = [r[0] for r in mapping_rows if r and str(r[0]).strip()]

    return (
        "Members loaded. Map each person to a Splitwise member, then submit.",
        gr.update(value=member_rows),
        member_state,
        gr.update(choices=member_names, value=payer_default),
        gr.update(value=mapping_rows),
        gr.update(choices=people_choices, value=(people_choices[0] if people_choices else None)),
        gr.update(choices=member_names, value=(member_names[0] if member_names else None)),
    )


def do_splitwise_load_members(
    api_key: str,
    selected_group: str,
    group_map: Dict[str, int],
    splitwise_rows: List[List[str]],
    current_user_id: int,
):
    return _load_splitwise_group_members(
        api_key=api_key,
        selected_group=selected_group,
        group_map=group_map,
        mapping_source_rows=splitwise_rows or [],
        current_user_id=current_user_id,
    )


def do_seed_manual_splitwise_rows(people_csv: str, mapping_df: Any):
    people: List[str] = []
    seen = set()
    for raw in (people_csv or "").split(","):
        person = raw.strip()
        if person and person not in seen:
            seen.add(person)
            people.append(person)

    existing_rows = _table_to_rows(mapping_df)
    existing_by_person: Dict[str, List[str]] = {}
    for row in existing_rows:
        if not row:
            continue
        person = str(row[0]).strip() if len(row) > 0 else ""
        if not person:
            continue
        amount = str(row[1]).strip() if len(row) > 1 else ""
        mapped = str(row[2]).strip() if len(row) > 2 else ""
        existing_by_person[person] = [amount, mapped]

    if not people:
        choices = [str(r[0]).strip() for r in existing_rows if r and str(r[0]).strip()]
        return (
            gr.update(value=existing_rows),
            gr.update(choices=choices, value=(choices[0] if choices else None)),
            "Enter at least one person name.",
        )

    rows: List[List[str]] = []
    for person in people:
        amount, mapped = existing_by_person.get(person, ["", ""])
        rows.append([person, amount, mapped])

    return (
        gr.update(value=rows),
        gr.update(choices=people, value=(people[0] if people else None)),
        "Manual rows ready. Enter amounts and map each person to a Splitwise member.",
    )


def do_splitwise_load_members_manual(
    api_key: str,
    selected_group: str,
    group_map: Dict[str, int],
    mapping_df: Any,
    current_user_id: int,
):
    return _load_splitwise_group_members(
        api_key=api_key,
        selected_group=selected_group,
        group_map=group_map,
        mapping_source_rows=_table_to_rows(mapping_df),
        current_user_id=current_user_id,
    )


def do_splitwise_load_members_direct(
    api_key: str,
    selected_group: str,
    group_map: Dict[str, int],
    current_user_id: int,
):
    if not api_key:
        return (
            "Connect Splitwise first.",
            gr.update(value=[]),
            [],
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(value=[]),
        )
    if not selected_group or selected_group not in group_map:
        return (
            "Select a valid group.",
            gr.update(value=[]),
            [],
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(value=[]),
        )

    try:
        s = _get_splitwise_client(api_key)
        group = s.getGroup(int(group_map[selected_group]))
        members = group.getMembers() if hasattr(group, "getMembers") else []
        if not members and hasattr(group, "getUsers"):
            members = group.getUsers()
    except Exception as e:
        return (
            f"Failed to load members: {type(e).__name__}: {e}",
            gr.update(value=[]),
            [],
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(value=[]),
        )

    member_rows: List[List[str]] = []
    member_state: List[Dict[str, Any]] = []
    member_names: List[str] = []
    for m in members or []:
        name = _member_display_name(m)
        uid = int(m.getId())
        member_rows.append([name, str(uid)])
        member_state.append({"id": uid, "name": name})
        member_names.append(name)

    payer_default = member_names[0] if member_names else None
    for m in member_state:
        if m["id"] == current_user_id:
            payer_default = m["name"]
            break

    return (
        "Members loaded. Select a member, enter an amount, and add it to the expense.",
        gr.update(value=member_rows),
        member_state,
        gr.update(choices=member_names, value=payer_default),
        gr.update(choices=member_names, value=(member_names[0] if member_names else None)),
        gr.update(value=[]),
    )


def _parse_decimal(value: Any) -> Decimal:
    text = str(value).strip()
    if not text:
        return Decimal("0")
    return Decimal(text)


def _table_to_rows(table: Any) -> List[List[Any]]:
    if table is None:
        return []
    if isinstance(table, dict):
        data = table.get("data")
        return data if isinstance(data, list) else []
    if isinstance(table, list):
        return table
    if hasattr(table, "values"):
        try:
            return table.values.tolist()
        except Exception:
            return []
    return []


def do_splitwise_submit(
    sw_api_key: str,
    selected_group: str,
    group_map: Dict[str, int],
    member_state: List[Dict[str, Any]],
    mapping_df: Any,
    payer_name: str,
    description: str,
) -> str:
    if not sw_api_key:
        return "Connect Splitwise first."
    if not selected_group or selected_group not in group_map:
        return "Select a valid group."
    if not member_state:
        return "Load group members first."
    if not payer_name:
        return "Select who paid."

    name_to_id = {m["name"]: int(m["id"]) for m in member_state}
    if payer_name not in name_to_id:
        return "Payer must be one of the loaded members."
    payer_id = name_to_id[payer_name]

    rows = _table_to_rows(mapping_df)
    owed_by_id: Dict[int, Decimal] = {}
    total = Decimal("0")

    try:
        for r in rows:
            if not r or len(r) < 3:
                continue
            amount = _parse_decimal(r[1])
            mapped_member = str(r[2]).strip()
            if amount < 0:
                return "Amounts cannot be negative."
            if amount == 0:
                continue
            if mapped_member not in name_to_id:
                return f"Invalid member mapping: '{mapped_member}'."
            uid = name_to_id[mapped_member]
            owed_by_id[uid] = owed_by_id.get(uid, Decimal("0")) + amount
            total += amount
    except InvalidOperation:
        return "Invalid amount format. Use numbers like 120 or 120.50."

    if total <= 0:
        return "Enter at least one amount > 0."

    expense = Expense()
    expense.setDescription((description or "").strip() or "Receipt split")
    expense.setGroupId(int(group_map[selected_group]))
    expense.setCost(format(total, "f"))

    users: List[ExpenseUser] = []
    payer_row: Optional[ExpenseUser] = None
    for m in member_state:
        uid = int(m["id"])
        owed = owed_by_id.get(uid, Decimal("0"))
        eu = ExpenseUser()
        eu.setId(uid)
        eu.setOwedShare(format(owed, "f"))
        eu.setPaidShare("0")
        if uid == payer_id:
            payer_row = eu
        users.append(eu)

    if payer_row is None:
        payer_row = ExpenseUser()
        payer_row.setId(payer_id)
        payer_row.setOwedShare("0")
        users.append(payer_row)
    payer_row.setPaidShare(format(total, "f"))

    expense.setUsers(users)

    try:
        s = _get_splitwise_client(sw_api_key)
        result = s.createExpense(expense)
        if isinstance(result, tuple):
            created, errors = result
        else:
            created, errors = result, None
        if errors:
            return f"Splitwise rejected expense: {errors}"
        expense_id = created.getId() if created else None
        return f"Expense created successfully. Expense ID: {expense_id}"
    except Exception as e:
        return f"Failed to create expense: {type(e).__name__}: {e}"


def do_set_member_mapping(mapping_df: Any, person_name: str, member_name: str):
    rows = _table_to_rows(mapping_df)
    if not rows:
        return gr.update(value=[]), "No rows to map."
    if not person_name:
        return gr.update(value=rows), "Select a person to map."
    if not member_name:
        return gr.update(value=rows), "Select a Splitwise member."

    updated = False
    for r in rows:
        if not r:
            continue
        person = str(r[0]).strip() if len(r) > 0 else ""
        if person == person_name:
            while len(r) < 3:
                r.append("")
            r[2] = member_name
            updated = True
            break

    if not updated:
        return gr.update(value=rows), f"Person '{person_name}' not found in mapping table."
    return gr.update(value=rows), f"Mapped '{person_name}' -> '{member_name}'."


def do_add_splitwise_member_amount(table: Any, member_name: str, amount_text: str):
    rows = _table_to_rows(table)
    if not member_name:
        return gr.update(value=rows), "", "Select a Splitwise member."

    try:
        amount = _parse_decimal(amount_text)
    except InvalidOperation:
        return gr.update(value=rows), amount_text, "Invalid amount format. Use numbers like 120 or 120.50."

    if amount <= 0:
        return gr.update(value=rows), amount_text, "Enter an amount greater than 0."

    updated = False
    for row in rows:
        if not row:
            continue
        name = str(row[0]).strip() if len(row) > 0 else ""
        if name != member_name:
            continue
        existing = Decimal("0")
        try:
            existing = _parse_decimal(row[1] if len(row) > 1 else "0")
        except InvalidOperation:
            existing = Decimal("0")
        while len(row) < 2:
            row.append("")
        row[1] = format(existing + amount, "f")
        updated = True
        break

    if not updated:
        rows.append([member_name, format(amount, "f")])

    return gr.update(value=rows), "", f"Added {format(amount, 'f')} for {member_name}."


def do_remove_splitwise_member_amount(table: Any, member_name: str):
    rows = _table_to_rows(table)
    if not member_name:
        return gr.update(value=rows), "Select a Splitwise member to remove."

    filtered = [row for row in rows if row and str(row[0]).strip() != member_name]
    if len(filtered) == len(rows):
        return gr.update(value=rows), f"No amount row found for {member_name}."
    return gr.update(value=filtered), f"Removed amount row for {member_name}."


def do_splitwise_submit_direct(
    sw_api_key: str,
    selected_group: str,
    group_map: Dict[str, int],
    member_state: List[Dict[str, Any]],
    member_amounts_df: Any,
    payer_name: str,
    description: str,
) -> str:
    if not sw_api_key:
        return "Connect Splitwise first."
    if not selected_group or selected_group not in group_map:
        return "Select a valid group."
    if not member_state:
        return "Load group members first."
    if not payer_name:
        return "Select who paid."

    name_to_id = {m["name"]: int(m["id"]) for m in member_state}
    if payer_name not in name_to_id:
        return "Payer must be one of the loaded members."
    payer_id = name_to_id[payer_name]

    rows = _table_to_rows(member_amounts_df)
    owed_by_id: Dict[int, Decimal] = {}
    total = Decimal("0")

    try:
        for r in rows:
            if not r or len(r) < 2:
                continue
            member_name = str(r[0]).strip()
            amount = _parse_decimal(r[1])
            if amount < 0:
                return "Amounts cannot be negative."
            if amount == 0:
                continue
            if member_name not in name_to_id:
                return f"Invalid member entry: '{member_name}'."
            uid = name_to_id[member_name]
            owed_by_id[uid] = owed_by_id.get(uid, Decimal("0")) + amount
            total += amount
    except InvalidOperation:
        return "Invalid amount format. Use numbers like 120 or 120.50."

    if total <= 0:
        return "Enter at least one amount > 0."

    expense = Expense()
    expense.setDescription((description or "").strip() or "Manual split")
    expense.setGroupId(int(group_map[selected_group]))
    expense.setCost(format(total, "f"))

    users: List[ExpenseUser] = []
    payer_row: Optional[ExpenseUser] = None
    for m in member_state:
        uid = int(m["id"])
        owed = owed_by_id.get(uid, Decimal("0"))
        eu = ExpenseUser()
        eu.setId(uid)
        eu.setOwedShare(format(owed, "f"))
        eu.setPaidShare("0")
        if uid == payer_id:
            payer_row = eu
        users.append(eu)

    if payer_row is None:
        payer_row = ExpenseUser()
        payer_row.setId(payer_id)
        payer_row.setOwedShare("0")
        users.append(payer_row)
    payer_row.setPaidShare(format(total, "f"))

    expense.setUsers(users)

    try:
        s = _get_splitwise_client(sw_api_key)
        result = s.createExpense(expense)
        if isinstance(result, tuple):
            created, errors = result
        else:
            created, errors = result, None
        if errors:
            return f"Splitwise rejected expense: {errors}"
        expense_id = created.getId() if created else None
        return f"Expense created successfully. Expense ID: {expense_id}"
    except Exception as e:
        return f"Failed to create expense: {type(e).__name__}: {e}"


def do_splitwise_only_login(username: str, password: str):
    expected_username = settings.splitwise_only_username.strip()
    expected_password = settings.splitwise_only_password

    if not expected_username or not expected_password:
        gr.Warning("Splitwise-only login is not configured. Set SPLITWISE_ONLY_USERNAME and SPLITWISE_ONLY_PASSWORD in .env.")
        return gr.update(visible=True), gr.update(visible=False), gr.update(value=username), gr.update(value=""), True, gr.update(visible=False, value="")

    if not username or not password:
        gr.Warning("Enter both username and password.")
        return gr.update(visible=True), gr.update(visible=False), gr.update(value=username), gr.update(value=""), True, gr.update(visible=False, value="")

    if username.strip() != expected_username or password != expected_password:
        gr.Warning("Invalid username or password.")
        return gr.update(visible=True), gr.update(visible=False), gr.update(value=username), gr.update(value=""), True, gr.update(visible=False, value="")

    gr.Info("Login successful.")
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value=""),
        gr.update(value=""),
        True,
        gr.update(visible=False, value=""),
    )


def do_splitwise_only_skip_login():
    gr.Info("Login skipped. Enter your Splitwise API key manually.")
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value=""),
        gr.update(value=""),
        False,
        gr.update(visible=True, value=""),
    )


def do_splitwise_only_connect(use_env_key: bool, manual_api_key: str):
    api_key = settings.splitwise_api_key.strip() if use_env_key else (manual_api_key or "").strip()
    if not api_key:
        if use_env_key:
            return (
                "Splitwise API key is not configured. Set SPLITWISE_API_KEY in .env.",
                "",
                gr.update(choices=[], value=None),
                {},
                0,
            )
        return (
            "Enter a Splitwise API key.",
            "",
            gr.update(choices=[], value=None),
            {},
            0,
        )
    return do_splitwise_connect(api_key)


def do_splitwise_only_logout():
    gr.Info("Logged out.")
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(value=""),
        True,
        gr.update(visible=False, value=""),
        "",
        {},
        [],
        0,
        gr.update(choices=[], value=None),
        gr.update(value=[]),
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
        gr.update(value=[]),
        gr.update(value=""),
        gr.update(value="Manual split"),
        gr.update(value=""),
        gr.update(value=""),
    )


# -------- UI --------


def build_ui():
    theme = gr.themes.Soft(primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="gray").set(
    body_background_fill="white",
    body_text_color="black"
    )
    with gr.Blocks(title="Bill Splitter (HF Spaces)", theme=theme) as demo:
        gr.Markdown("# 💰 Bill Splitter")
        state_api_key = gr.State("")
        state_receipt_id = gr.State("")
        state_alloc_line_ids = gr.State([])
        state_splitwise_rows = gr.State([])
        state_sw_api_key = gr.State("")
        state_sw_group_map = gr.State({})
        state_sw_members = gr.State([])
        state_sw_current_user_id = gr.State(0)
        state_sw_only_api_key = gr.State("")
        state_sw_only_group_map = gr.State({})
        state_sw_only_members = gr.State([])
        state_sw_only_current_user_id = gr.State(0)
        state_sw_only_use_env_key = gr.State(True)

        with gr.Tabs():
            with gr.Tab("Receipt Splitter"):
                gr.Markdown("Use Gemini API Key, Upload Receipts and Share amounts with friends")

                with gr.Row():
                    model = gr.Textbox(label="Gemini Model", value=DEFAULT_MODEL)

                gr.Markdown("## Gemini API Key")

                with gr.Row():
                    api_key_in = gr.Textbox(label="Gemini API Key", type="password", placeholder="Paste your API key here")
                with gr.Row():
                    #btn_test = gr.Button("Test Key")
                    btn_save = gr.Button("Save & Continue", variant="primary")

                key_status = gr.Textbox(label="Key Status", value="", lines=3)

                gr.Markdown("## Receipt Extraction & Splitting")

                with gr.Group(visible=False) as main_group:
                    with gr.Row():
                        receipt_file = gr.File(
                            label="Upload receipt (PDF/JPG/PNG/WebP)",
                            file_types=[".pdf", ".jpg", ".jpeg", ".png", ".webp"],
                        )
                        btn_extract = gr.Button("Extract Bill", variant="primary")

                    extract_status = gr.Textbox(label="Extraction Status", lines=6)

                    line_items_df = gr.Dataframe(
                        headers=["id", "description", "amount", "type"],
                        datatype=["str", "str", "str", "str"],
                        row_count=10,
                        col_count=(4, "fixed"),
                        interactive=True,
                        label="Line Items (editable)",
                    )

                    with gr.Row():
                        people_csv = gr.Textbox(
                            label="People (comma-separated)",
                            placeholder="e.g. Alice, Bob, Cara",
                        )
                        btn_build_alloc = gr.Button("Build/Reset Allocation Matrix", variant="primary")

                    gr.Markdown("### Allocation Matrix (weights)\nBlank = 0 share. Check equal to fill 1 for every person on that item.")

                    alloc_df = gr.Dataframe(
                        headers=["Item", "Price", "Equal Shares"],
                        datatype=["str", "str", "bool"],
                        row_count=1,
                        col_count=(3, "dynamic"),
                        interactive=True,
                        label="Allocations",
                    )

                    btn_compute = gr.Button("Compute Totals", variant="primary")
                    compute_out = gr.Textbox(label="Result", lines=14)

                    btn_add_splitwise = gr.Button("Add to Splitwise", variant="primary")

                    with gr.Group(visible=False) as splitwise_group:
                        gr.Markdown("## Splitwise")
                        sw_status = gr.Textbox(label="Splitwise Status", lines=3)
                        sw_api_key_in = gr.Textbox(label="Splitwise API Key", type="password")
                        btn_sw_connect = gr.Button("Connect Splitwise")
                        sw_group_dropdown = gr.Dropdown(label="Group", choices=[])
                        btn_sw_load_members = gr.Button("Load Group Members")
                        sw_members_table = gr.Dataframe(
                            headers=["Member", "Splitwise User ID"],
                            datatype=["str", "str"],
                            interactive=False,
                            label="Group Members",
                        )
                        sw_mapping_df = gr.Dataframe(
                            headers=["Person", "Amount", "Splitwise Member"],
                            datatype=["str", "str", "str"],
                            interactive=True,
                            label="Map Person Amounts to Splitwise Members",
                        )
                        with gr.Row():
                            sw_map_person = gr.Dropdown(label="Person", choices=[])
                            sw_map_member = gr.Dropdown(label="Splitwise Member", choices=[])
                            btn_sw_set_map = gr.Button("Set Mapping", variant="primary")
                        sw_payer = gr.Dropdown(label="Who Paid?", choices=[])
                        sw_description = gr.Textbox(label="Expense Description", value="Receipt split")
                        btn_sw_submit = gr.Button("Submit to Splitwise", variant="primary")
                        sw_submit_out = gr.Textbox(label="Splitwise Submit Result", lines=3)

            with gr.Tab("Splitwise Only"):
                gr.Markdown("Login first, then connect to Splitwise")
                with gr.Group(visible=True) as sw_only_login_group:
                    sw_only_username_in = gr.Textbox(label="Username")
                    sw_only_password_in = gr.Textbox(label="Password", type="password")
                    with gr.Row():
                        btn_sw_only_login = gr.Button("🔐 Login", variant="primary")
                        btn_sw_only_skip_login = gr.Button("⏭️ Skip", variant="secondary")
                with gr.Group(visible=False) as sw_only_main_group:
                    sw_only_status = gr.Textbox(label="Splitwise Status", lines=3)
                    sw_only_api_key_manual_in = gr.Textbox(label="Splitwise API Key", type="password", visible=False)
                    btn_sw_only_connect = gr.Button("Connect Splitwise")
                    sw_only_group_dropdown = gr.Dropdown(label="Group", choices=[])
                    btn_sw_only_load_members = gr.Button("Load Group Members")
                    sw_only_members_table = gr.Dataframe(
                        headers=["Member", "Splitwise User ID"],
                        datatype=["str", "str"],
                        interactive=False,
                        label="Group Members",
                    )
                    with gr.Row():
                        sw_only_member_select = gr.Dropdown(label="Member", choices=[])
                        sw_only_amount_in = gr.Textbox(label="Amount", placeholder="e.g. 24.50")
                        btn_sw_only_add_amount = gr.Button("Add Amount")
                    btn_sw_only_remove_amount = gr.Button("Remove Selected Member Row")
                    sw_only_amounts_df = gr.Dataframe(
                        headers=["Member", "Amount"],
                        datatype=["str", "str"],
                        row_count=8,
                        col_count=(2, "fixed"),
                        interactive=True,
                        label="Member Amounts",
                    )
                    sw_only_payer = gr.Dropdown(label="Who Paid?", choices=[])
                    sw_only_description = gr.Textbox(label="Expense Description", value="Split")
                    btn_sw_only_submit = gr.Button("📤 Submit to Splitwise")
                    sw_only_submit_out = gr.Textbox(label="Splitwise Submit Result", lines=3)
                    btn_sw_only_logout = gr.Button("🚪 Logout")

        #btn_test.click(do_test_key, inputs=[api_key_in, model], outputs=[key_status])

        btn_save.click(
            do_save_key,
            inputs=[api_key_in, model],
            outputs=[state_api_key, key_status, main_group],
        )

        def _extract_wrap(api_key: str, model: str, file_obj: Any):
            status, rid, rows, reset_ids = do_extract(api_key, model, file_obj)
            return status, rid, gr.update(value=rows), reset_ids

        btn_extract.click(
            _extract_wrap,
            inputs=[state_api_key, model, receipt_file],
            outputs=[extract_status, state_receipt_id, line_items_df, state_alloc_line_ids],
        )

        btn_build_alloc.click(
            do_build_alloc,
            inputs=[people_csv, line_items_df],
            outputs=[alloc_df, state_alloc_line_ids],
        )

        alloc_df.change(
            do_apply_equal_shares,
            inputs=[alloc_df, people_csv],
            outputs=[alloc_df],
        )

        btn_compute.click(
            do_compute_with_splitwise,
            inputs=[state_receipt_id, people_csv, line_items_df, alloc_df, state_alloc_line_ids],
            outputs=[compute_out, state_splitwise_rows],
        )

        btn_add_splitwise.click(
            do_open_splitwise,
            inputs=[state_splitwise_rows],
            outputs=[splitwise_group, sw_status, sw_mapping_df, sw_map_person],
        )

        btn_sw_connect.click(
            do_splitwise_connect,
            inputs=[sw_api_key_in],
            outputs=[sw_status, state_sw_api_key, sw_group_dropdown, state_sw_group_map, state_sw_current_user_id],
        )

        btn_sw_load_members.click(
            do_splitwise_load_members,
            inputs=[state_sw_api_key, sw_group_dropdown, state_sw_group_map, state_splitwise_rows, state_sw_current_user_id],
            outputs=[sw_status, sw_members_table, state_sw_members, sw_payer, sw_mapping_df, sw_map_person, sw_map_member],
        )

        btn_sw_set_map.click(
            do_set_member_mapping,
            inputs=[sw_mapping_df, sw_map_person, sw_map_member],
            outputs=[sw_mapping_df, sw_status],
        )

        btn_sw_submit.click(
            do_splitwise_submit,
            inputs=[
                state_sw_api_key,
                sw_group_dropdown,
                state_sw_group_map,
                state_sw_members,
                sw_mapping_df,
                sw_payer,
                sw_description,
            ],
            outputs=[sw_submit_out],
        )

        btn_sw_only_login.click(
            do_splitwise_only_login,
            inputs=[sw_only_username_in, sw_only_password_in],
            outputs=[sw_only_login_group, sw_only_main_group, sw_only_username_in, sw_only_password_in, state_sw_only_use_env_key, sw_only_api_key_manual_in],
        )

        btn_sw_only_skip_login.click(
            do_splitwise_only_skip_login,
            outputs=[sw_only_login_group, sw_only_main_group, sw_only_username_in, sw_only_password_in, state_sw_only_use_env_key, sw_only_api_key_manual_in],
        )

        btn_sw_only_connect.click(
            do_splitwise_only_connect,
            inputs=[state_sw_only_use_env_key, sw_only_api_key_manual_in],
            outputs=[sw_only_status, state_sw_only_api_key, sw_only_group_dropdown, state_sw_only_group_map, state_sw_only_current_user_id],
        )

        btn_sw_only_load_members.click(
            do_splitwise_load_members_direct,
            inputs=[state_sw_only_api_key, sw_only_group_dropdown, state_sw_only_group_map, state_sw_only_current_user_id],
            outputs=[sw_only_status, sw_only_members_table, state_sw_only_members, sw_only_payer, sw_only_member_select, sw_only_amounts_df],
        )

        btn_sw_only_add_amount.click(
            do_add_splitwise_member_amount,
            inputs=[sw_only_amounts_df, sw_only_member_select, sw_only_amount_in],
            outputs=[sw_only_amounts_df, sw_only_amount_in, sw_only_status],
        )

        btn_sw_only_remove_amount.click(
            do_remove_splitwise_member_amount,
            inputs=[sw_only_amounts_df, sw_only_member_select],
            outputs=[sw_only_amounts_df, sw_only_status],
        )

        btn_sw_only_submit.click(
            do_splitwise_submit_direct,
            inputs=[
                state_sw_only_api_key,
                sw_only_group_dropdown,
                state_sw_only_group_map,
                state_sw_only_members,
                sw_only_amounts_df,
                sw_only_payer,
                sw_only_description,
            ],
            outputs=[sw_only_submit_out],
        )

        btn_sw_only_logout.click(
            do_splitwise_only_logout,
            outputs=[
                sw_only_login_group,
                sw_only_main_group,
                sw_only_username_in,
                sw_only_password_in,
                state_sw_only_use_env_key,
                sw_only_api_key_manual_in,
                state_sw_only_api_key,
                state_sw_only_group_map,
                state_sw_only_members,
                state_sw_only_current_user_id,
                sw_only_group_dropdown,
                sw_only_members_table,
                sw_only_payer,
                sw_only_member_select,
                sw_only_amounts_df,
                sw_only_amount_in,
                sw_only_description,
                sw_only_status,
                sw_only_submit_out,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()

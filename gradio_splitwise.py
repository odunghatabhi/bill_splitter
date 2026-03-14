from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Tuple

import gradio as gr
from splitwise import Splitwise
from splitwise.expense import Expense
from splitwise.user import ExpenseUser

from bill_splitter import splitwise_app


def _get_client(api_key: str) -> Splitwise:
    return Splitwise(
        splitwise_app.CONSUMER_KEY,
        splitwise_app.CONSUMER_SECRET,
        api_key=api_key,
    )


def _member_name(u: Any) -> str:
    first = (u.getFirstName() or "").strip()
    last = (u.getLastName() or "").strip()
    full = f"{first} {last}".strip()
    return full or f"User {u.getId()}"


def connect_and_fetch_groups(api_key: str) -> Tuple[str, str, gr.update, Dict[str, int], int]:
    if not api_key or not api_key.strip():
        return "Enter Splitwise API key.", "", gr.update(choices=[], value=None), {}, 0

    api_key = api_key.strip()
    try:
        client = _get_client(api_key)
        current = client.getCurrentUser()
        current_id = int(current.getId()) if current else 0
        groups = client.getGroups()
    except Exception as exc:
        return f"Connection failed: {type(exc).__name__}: {exc}", "", gr.update(choices=[], value=None), {}, 0

    if not groups:
        return "Connected. No groups found.", api_key, gr.update(choices=[], value=None), {}, current_id

    group_map: Dict[str, int] = {}
    for g in groups:
        label = f"{g.getName()} ({g.getId()})"
        group_map[label] = int(g.getId())

    choices = list(group_map.keys())
    return (
        "Connected. Select a group.",
        api_key,
        gr.update(choices=choices, value=choices[0]),
        group_map,
        current_id,
    )


def load_group_members(
    api_key: str,
    group_label: str,
    group_map: Dict[str, int],
    current_user_id: int,
) -> Tuple[str, gr.update, List[Dict[str, Any]], gr.update]:
    if not api_key:
        return "Connect first.", gr.update(value=[]), [], gr.update(choices=[], value=None)
    if not group_label or group_label not in group_map:
        return "Select a valid group.", gr.update(value=[]), [], gr.update(choices=[], value=None)

    try:
        client = _get_client(api_key)
        group_id = group_map[group_label]
        group = client.getGroup(group_id)
        users = group.getMembers() if hasattr(group, "getMembers") else []
        if not users and hasattr(group, "getUsers"):
            users = group.getUsers()
    except Exception as exc:
        return (
            f"Failed loading members: {type(exc).__name__}: {exc}",
            gr.update(value=[]),
            [],
            gr.update(choices=[], value=None),
        )

    members: List[Dict[str, Any]] = []
    for u in users:
        members.append({"id": int(u.getId()), "name": _member_name(u)})

    rows = [[m["name"], ""] for m in members]
    payer_choices = [m["name"] for m in members]
    payer_default = payer_choices[0] if payer_choices else None
    if current_user_id:
        for m in members:
            if m["id"] == current_user_id:
                payer_default = m["name"]
                break

    return (
        "Members loaded. Enter owed amount per member.",
        gr.update(value=rows),
        members,
        gr.update(choices=payer_choices, value=payer_default),
    )


def _to_decimal(value: Any) -> Decimal:
    text = str(value).strip()
    if not text:
        return Decimal("0")
    return Decimal(text)


def submit_expense(
    api_key: str,
    group_label: str,
    group_map: Dict[str, int],
    members_state: List[Dict[str, Any]],
    member_amounts_table: Any,
    payer_name: str,
    description: str,
) -> str:
    if not api_key:
        return "Connect first."
    if not group_label or group_label not in group_map:
        return "Select a valid group."
    if not members_state:
        return "Load members first."
    if not payer_name:
        return "Select who paid."

    name_to_id = {m["name"]: int(m["id"]) for m in members_state}
    if payer_name not in name_to_id:
        return "Selected payer is not in this group."
    payer_id = name_to_id[payer_name]

    rows = member_amounts_table.get("data", []) if isinstance(member_amounts_table, dict) else (member_amounts_table or [])
    owed_by_id: Dict[int, Decimal] = {}
    total = Decimal("0")

    try:
        for row in rows:
            if not row or len(row) < 2:
                continue
            name = str(row[0]).strip()
            if name not in name_to_id:
                continue
            amount = _to_decimal(row[1])
            if amount < 0:
                return f"Amount for {name} cannot be negative."
            if amount == 0:
                continue
            uid = name_to_id[name]
            owed_by_id[uid] = owed_by_id.get(uid, Decimal("0")) + amount
            total += amount
    except InvalidOperation:
        return "Invalid amount format. Use numbers like 120 or 120.50."

    if total <= 0:
        return "Enter at least one amount > 0."

    expense = Expense()
    expense.setDescription((description or "").strip() or "Group expense")
    expense.setGroupId(int(group_map[group_label]))
    expense.setCost(format(total, "f"))

    users: List[ExpenseUser] = []
    payer_row: ExpenseUser | None = None
    for member in members_state:
        uid = int(member["id"])
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
        client = _get_client(api_key)
        result = client.createExpense(expense)
        if isinstance(result, tuple):
            created, errors = result
        else:
            created, errors = result, None

        if errors:
            return f"Splitwise rejected expense: {errors}"

        expense_id = created.getId() if created else None
        return f"Expense created successfully. Expense ID: {expense_id}"
    except Exception as exc:
        return f"Failed to create expense: {type(exc).__name__}: {exc}"


def build_ui():
    with gr.Blocks(title="Splitwise Group Expense") as demo:
        gr.Markdown("# Splitwise Group Expense")
        gr.Markdown("Enter Splitwise API key, select group, enter member amounts, and submit.")

        state_api_key = gr.State("")
        state_group_map = gr.State({})
        state_members = gr.State([])
        state_current_user_id = gr.State(0)

        api_key = gr.Textbox(label="Splitwise API Key", type="password")
        connect_btn = gr.Button("Connect & Fetch Groups")
        status = gr.Textbox(label="Status", lines=3)

        group_dropdown = gr.Dropdown(label="Select Group", choices=[])
        load_members_btn = gr.Button("Load Members")

        member_amounts = gr.Dataframe(
            headers=["Member", "Amount"],
            datatype=["str", "str"],
            interactive=True,
            label="Members and Owed Amounts",
        )

        payer_dropdown = gr.Dropdown(label="Who Paid?", choices=[])
        description = gr.Textbox(label="Expense Description", value="Group expense")
        submit_btn = gr.Button("Submit to Splitwise")
        submit_status = gr.Textbox(label="Submit Status", lines=4)

        connect_btn.click(
            connect_and_fetch_groups,
            inputs=[api_key],
            outputs=[status, state_api_key, group_dropdown, state_group_map, state_current_user_id],
        )

        load_members_btn.click(
            load_group_members,
            inputs=[state_api_key, group_dropdown, state_group_map, state_current_user_id],
            outputs=[status, member_amounts, state_members, payer_dropdown],
        )

        submit_btn.click(
            submit_expense,
            inputs=[
                state_api_key,
                group_dropdown,
                state_group_map,
                state_members,
                member_amounts,
                payer_dropdown,
                description,
            ],
            outputs=[submit_status],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch()

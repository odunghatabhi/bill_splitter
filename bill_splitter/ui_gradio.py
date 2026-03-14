# ui_gradio_local.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os
import uuid

import gradio as gr
import requests

from bill_splitter import splitwise_app  # your splitwise.py module

API_BASE_DEFAULT = "http://127.0.0.1:8000"
DEFAULT_MODEL = "gemini-3-flash-preview"

# ------------------ Helpers ------------------

def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _normalize_line_items_df(df: Any) -> List[Dict[str, Any]]:
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
        row = {h: r[i] if i < len(r) else None for i, h in enumerate(headers)}
        li_id = _safe_str(row.get("id")).strip()
        desc = _safe_str(row.get("description")).strip() or "(no description)"
        amt = _safe_str(row.get("amount")).strip() or "0"
        typ = _safe_str(row.get("type")).strip().lower() or "item"
        if li_id:
            items.append({"id": li_id, "description": desc, "amount": amt, "type": typ})
    return items


def _build_allocation_matrix(line_items: List[Dict[str, Any]], people: List[str]):
    people = [p.strip() for p in people if p and p.strip()]
    headers = ["item"] + people
    rows = [[li["description"]] + [""]*len(people) for li in line_items]
    line_ids = [li["id"] for li in line_items]
    return rows, headers, line_ids


def _allocations_df_to_dict(alloc_df: Any, people: List[str], line_ids: List[str]) -> Dict[str, Dict[str, str]]:
    if alloc_df is None:
        return {}
    people = [p.strip() for p in people if p.strip()]
    if not people or not line_ids:
        return {}

    if isinstance(alloc_df, dict) and "data" in alloc_df:
        rows = alloc_df["data"]
    elif isinstance(alloc_df, list):
        rows = alloc_df
    else:
        try:
            rows = alloc_df.values.tolist()
        except Exception:
            return {}

    allocations: Dict[str, Dict[str, str]] = {}
    for row_idx, r in enumerate(rows):
        if row_idx >= len(line_ids):
            break
        line_id = line_ids[row_idx]
        per = {}
        for i, person in enumerate(people, start=1):
            if i < len(r):
                val = _safe_str(r[i]).strip()
                if val:
                    per[person] = val
        if per:
            allocations[line_id] = per
    return allocations

# ------------------ API Calls ------------------

def api_test_key(api_base: str, api_key: str, model: str) -> str:
    if not api_key or not api_key.strip():
        return "Enter an API key first."
    try:
        url = f"{api_base.rstrip('/')}/test-key"
        r = requests.post(url, data={"api_key": api_key, "model": model}, timeout=30)
        r.raise_for_status()
        data = r.json()
        return f"{'✅' if data.get('ok') else '❌'} {data.get('message','')}"
    except Exception as e:
        return f"❌ Test failed: {type(e).__name__}: {e}"


def api_extract(api_base: str, api_key: str, model: str, file_obj: Any) -> Tuple[str, Optional[str], List[List[Any]]]:
    if not api_key or not api_key.strip():
        return "❌ No API key saved.", None, []
    if file_obj is None:
        return "❌ Please upload a file.", None, []

    file_path = getattr(file_obj, "name", None) or str(file_obj)
    try:
        url = f"{api_base.rstrip('/')}/extract"
        with open(file_path, "rb") as f:
            files = {"file": (file_path.split("\\")[-1], f)}
            data = {"api_key": api_key, "model": model}
            r = requests.post(url, data=data, files=files, timeout=180)
        r.raise_for_status()
        receipt = r.json()
        rid = receipt.get("receipt_id")
        rows = [[li.get("id"), li.get("description"), str(li.get("amount")), li.get("type")]
                for li in receipt.get("line_items", [])]
        return "✅ Extracted line items.", rid, rows
    except Exception as e:
        return f"❌ Extraction failed: {type(e).__name__}: {e}", None, []


def api_compute(api_base: str, receipt_id: str, people_csv: str, line_items_df: Any, alloc_df: Any, line_ids: List[str]) -> str:
    if not receipt_id:
        return "❌ No receipt_id. Extract a receipt first."
    people = [p.strip() for p in (people_csv or "").split(",") if p.strip()]
    line_items = _normalize_line_items_df(line_items_df)
    allocations = _allocations_df_to_dict(alloc_df, people, line_ids)
    if not people or not line_items:
        return "❌ No people or line items"

    payload = {"receipt_id": receipt_id, "people": people, "line_items": line_items, "allocations": allocations,
               "rounding_decimals": 2}
    try:
        r = requests.post(f"{api_base.rstrip('/')}/compute", json=payload, timeout=60)
        r.raise_for_status()
        res = r.json()
        totals = res.get("totals", {})
        lines = ["✅ Totals per person:"]
        for p in people:
            lines.append(f"- {p}: {totals.get(p, '0.00')}")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Compute failed: {type(e).__name__}: {e}"

# ------------------ Gradio UI ------------------

def build_ui():
    with gr.Blocks(title="Bill Splitter + Splitwise") as demo:
        gr.Markdown("# Bill Splitter\nUpload receipts, split totals, push to Splitwise.")

        # ---- states ----
        state_api_key = gr.State("")
        state_receipt_id = gr.State("")
        state_alloc_line_ids = gr.State([])
        state_sw_oauth_secret = gr.State("")
        state_sw_access_token = gr.State("")
        state_sw_access_secret = gr.State("")
        state_sw_groups = gr.State([])
        state_sw_group_members = gr.State([])
        state_sw_group_id = gr.State(None)

        # ---- Gemini API key ----
        api_key_in = gr.Textbox(label="Gemini API Key", type="password")
        btn_test = gr.Button("Test Key")
        key_status = gr.Textbox(label="Key Status", lines=3)

        btn_test.click(lambda api: api_test_key(API_BASE_DEFAULT, api, "gemini-3-flash-preview"),
                       inputs=[api_key_in], outputs=[key_status])

        # ---- Receipt extraction ----
        receipt_file = gr.File(label="Upload receipt")
        btn_extract = gr.Button("Extract")
        extract_status = gr.Textbox(label="Extraction Status", lines=6)
        line_items_df = gr.Dataframe(headers=["id", "description", "amount", "type"], interactive=True)
        people_csv = gr.Textbox(label="People (comma-separated)")
        btn_build_alloc = gr.Button("Build Allocation Matrix")
        alloc_df = gr.Dataframe(headers=["item"], interactive=True)
        btn_compute = gr.Button("Compute Totals")
        compute_out = gr.Textbox(label="Result", lines=14)

        # ---- Splitwise Section ----
        gr.Markdown("## Push to Splitwise")
        btn_connect_sw = gr.Button("Connect to Splitwise")
        sw_auth_link = gr.Textbox(label="Authorization URL")
        btn_fetch_groups = gr.Button("Fetch Groups")
        group_dropdown = gr.Dropdown(label="Select Group", choices=[])
        sw_member_mapping = gr.Dataframe(headers=["Person", "Splitwise User"], interactive=True)
        btn_push_expense = gr.Button("Create Expense in Splitwise")
        sw_push_status = gr.Textbox(label="Splitwise Status", lines=5)
        expense_description = gr.Textbox(label="Expense Description", value="Expense for receipt")

        # ---- Handlers ----
        def on_connect_sw():
            url, secret = splitwise_app.get_authorize_url()
            return url, secret
        btn_connect_sw.click(on_connect_sw, inputs=[], outputs=[sw_auth_link, state_sw_oauth_secret])

        def on_fetch_groups(access_token, access_secret):
            groups = splitwise_app.get_groups(access_token, access_secret)
            choices = [f"{g.getName()} ({g.getId()})" for g in groups]
            return choices, groups
        btn_fetch_groups.click(on_fetch_groups,
                               inputs=[state_sw_access_token, state_sw_access_secret],
                               outputs=[group_dropdown, state_sw_groups])

        def on_group_select(choice, groups):
            if not choice or not groups:
                return gr.update(value=[]), [], None
            gid = int(choice.split("(")[-1].strip(")"))
            members = splitwise_app.get_group_members(state_sw_access_token.value,
                                                  state_sw_access_secret.value, gid)
            table = [[u["name"], ""] for u in members]
            return gr.update(value=table), members, gid
        group_dropdown.change(on_group_select, inputs=[group_dropdown, state_sw_groups],
                              outputs=[sw_member_mapping, state_sw_group_members, state_sw_group_id])

        def on_push_expense(description, member_table, group_id, totals_dict):
            expense_data = []
            member_map = {row[0]: row[1] for row in member_table if row[1]}
            for person, amount in totals_dict.items():
                sw_user = next((u["id"] for u in state_sw_group_members.value
                                if f"{u['name']}" == member_map.get(person)), None)
                if not sw_user:
                    continue
                expense_data.append({"user_id": sw_user, "paid_share": 0, "owed_share": float(amount)})
            try:
                res = splitwise_app.create_expense(state_sw_access_token.value, state_sw_access_secret.value,
                                               group_id, description, expense_data)
                return f"✅ Expense created. ID: {res.getId()}"
            except Exception as e:
                return f"❌ Failed: {type(e).__name__}: {e}"

        btn_push_expense.click(on_push_expense,
                       inputs=[expense_description, sw_member_mapping, state_sw_group_id, compute_out],
                       outputs=[sw_push_status])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
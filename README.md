---
metadata
title: Bill Splitter Splitwise
emoji: 🐠
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
---
# Bill Splitter

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-F97316)](https://www.gradio.app/)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![Splitwise](https://img.shields.io/badge/Splitwise-Expense%20Sharing-1DBF73)](https://www.splitwise.com/)
[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-Spaces-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/abhisheko97/Bill_Splitter_Splitwise)

Split receipts with Gemini, review extracted line items, assign item-level shares across a group, and compute per-person totals. The repo also includes a Splitwise submission flow for local use.

Live app: [Bill Splitter on Hugging Face Spaces](https://huggingface.co/spaces/abhisheko97/Bill_Splitter_Splitwise)

## Features

- Upload a receipt as PDF, JPG, PNG, or WebP
- Extract line items with Google Gemini
- Edit extracted rows before splitting
- Build a per-item allocation matrix for multiple people
- Compute final totals for each person
- Submit the split to Splitwise when running locally

## Splitwise availability

The Hugging Face Spaces app is currently for receipt extraction and bill splitting only.

The Splitwise feature is **not available right now on Hugging Face Spaces**. To use Splitwise, run the project locally with:

```bash
python app.py
```

## Tech stack

- Python
- Gradio
- Google Gemini via `google-genai`
- Splitwise SDK
- `python-dotenv`

## Project layout

```text
app.py                             Main Gradio entrypoint
bill_splitter/ui_gradio_spaces.py  UI used by the Spaces app
bill_splitter/gemini_client.py     Gemini receipt extraction
bill_splitter/splitter.py          Split computation logic
bill_splitter/splitwise_app.py     Splitwise helpers
gradio_splitwise.py                Standalone Splitwise-oriented UI
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file based on `.env.example`:

```env
SPLITWISE_CONSUMER_KEY=your_splitwise_consumer_key
SPLITWISE_CONSUMER_SECRET=your_splitwise_consumer_secret
SPLITWISE_API_KEY=
SPLITWISE_ACCESS_TOKEN=
SPLITWISE_ACCESS_TOKEN_SECRET=
```

Notes:

- Gemini API keys are entered directly in the UI
- Splitwise local support requires `SPLITWISE_CONSUMER_KEY` and `SPLITWISE_CONSUMER_SECRET`
- `GEMINI_MODEL` can be set in your environment and defaults to `gemini-3-flash-preview`

### 3. Run locally

```bash
python app.py
```

Open the local Gradio URL shown in the terminal.

## How to use

### Receipt extraction and bill splitting

1. Open the Hugging Face Space or run `python app.py` locally.
2. Paste your Gemini API key.
3. Upload a receipt file.
4. Click `Extract`.
5. Review and edit the line items if needed.
6. Enter people as a comma-separated list.
7. Build the allocation matrix.
8. Fill in item weights for each person.
9. Click `Compute Totals`.

### Splitwise flow

1. Run the app locally with `python app.py`.
2. Finish the extraction and total computation flow.
3. Click `Add to Splitwise`.
4. Enter your Splitwise API key.
5. Connect, choose a group, and load members.
6. Map each person to a Splitwise member.
7. Select who paid and submit the expense.

## Deployment

This repo includes a GitHub Actions workflow that syncs the app to Hugging Face Spaces.

- Space: [abhisheko97/Bill_Splitter_Splitwise](https://huggingface.co/spaces/abhisheko97/Bill_Splitter_Splitwise)
- Workflow: [deploy-space.yml](/c:/Users/abhis/Documents/GitHub/bill_splitter/.github/workflows/deploy-space.yml)

## Requirements

See [requirements.txt](/c:/Users/abhis/Documents/GitHub/bill_splitter/requirements.txt):

- `gradio`
- `pydantic`
- `python-dotenv`
- `google-genai`
- `splitwise`

from __future__ import annotations

import uuid
import mimetypes
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from bill_splitter.config import settings
from bill_splitter.models import ComputeRequest
from bill_splitter.splitter import compute_splits
from bill_splitter.gemini_client import extract_receipt, test_api_key
from bill_splitter.validate import validate_extraction

app = FastAPI(title="Bill Splitter API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/test-key")
def api_test_key(
    api_key: str = Form(...),
    model: str = Form(settings.default_model),
):
    ok, msg = test_api_key(api_key=api_key, model=model)
    return {"ok": ok, "message": msg}


@app.post("/extract")
async def api_extract(
    api_key: str = Form(...),
    model: str = Form(settings.default_model),
    receipt_id: str | None = Form(None),
    file: UploadFile = File(...),
):
    try:
        rid = receipt_id or f"r_{uuid.uuid4().hex[:12]}"

        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Enforce upload size limit
        max_bytes = int(settings.max_upload_mb) * 1024 * 1024
        if len(file_bytes) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max allowed is {settings.max_upload_mb} MB.",
            )

        # Robust MIME detection (works even if client doesn't send content-type)
        mime_type = file.content_type
        if not mime_type or mime_type == "application/octet-stream":
            guessed, _ = mimetypes.guess_type(file.filename or "")
            mime_type = guessed or "application/octet-stream"

        receipt = extract_receipt(
            api_key=api_key,
            model=model,
            file_bytes=file_bytes,
            mime_type=mime_type,
            receipt_id=rid,
        )

        receipt = validate_extraction(receipt)

        # IMPORTANT: jsonable_encoder makes Decimal JSON-serializable
        return JSONResponse(content=jsonable_encoder(receipt))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {type(e).__name__}: {e}")


@app.post("/compute")
def api_compute(req: ComputeRequest):
    try:
        result = compute_splits(req)
        return JSONResponse(content=jsonable_encoder(result))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compute failed: {type(e).__name__}: {e}")

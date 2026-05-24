"""
VQA FastAPI backend — two operating modes controlled by env vars:

  USE_LOCAL_MODEL=true   → load Salesforce/blip-vqa-base in-process (local dev)
  USE_LOCAL_MODEL=false  → proxy requests to HF Inference API    (Render deploy)

Required when USE_LOCAL_MODEL=false:  HF_TOKEN
Optional:                             HF_MODEL  (default: Salesforce/blip-vqa-base)
"""

import io
import os
import base64
import logging

from dotenv import load_dotenv
load_dotenv()

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

USE_LOCAL = os.environ.get("USE_LOCAL_MODEL", "false").lower() == "true"
HF_TOKEN  = os.environ.get("HF_TOKEN", "")
HF_MODEL  = os.environ.get("HF_MODEL", "Salesforce/blip-vqa-base")
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# ── local model (lazy-loaded on first request) ───────────────────────────────
_processor = None
_model     = None

def _get_local_model():
    global _processor, _model
    if _model is not None:
        return _processor, _model
    log.info("Loading BLIP model locally — this takes ~30 s on first run…")
    from transformers import BlipProcessor, BlipForQuestionAnswering
    _processor = BlipProcessor.from_pretrained(HF_MODEL)
    _model     = BlipForQuestionAnswering.from_pretrained(HF_MODEL)
    _model.eval()
    log.info("BLIP model ready.")
    return _processor, _model


# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="VQA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    image_b64: str
    question: str


class PredictResponse(BaseModel):
    answer: str
    question: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "local" if USE_LOCAL else "hf-api",
        "model": HF_MODEL,
        "token_set": bool(HF_TOKEN),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    # decode + validate image
    try:
        raw = base64.b64decode(req.image_b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image: {exc}")

    if USE_LOCAL:
        answer = _predict_local(img, req.question)
    else:
        answer = _predict_hf_api(img, req.question)

    return PredictResponse(answer=answer, question=req.question)


def _predict_local(img: Image.Image, question: str) -> str:
    import torch
    processor, model = _get_local_model()
    inputs = processor(img, question, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30)
    return processor.decode(out[0], skip_special_tokens=True)


def _predict_hf_api(img: Image.Image, question: str) -> str:
    if not HF_TOKEN:
        raise HTTPException(status_code=503, detail="HF_TOKEN env var not set on the server.")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"inputs": {"image": image_b64, "question": question}}

    try:
        resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="HuggingFace API timed out.")
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"HuggingFace API error: {exc}")

    if resp.status_code == 503:
        raise HTTPException(status_code=503, detail="Model is loading on HuggingFace, retry in ~20 s.")
    if not resp.ok:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    result = resp.json()
    if isinstance(result, list) and result:
        return result[0].get("answer", "")
    if isinstance(result, dict):
        return result.get("answer", "")
    return str(result)

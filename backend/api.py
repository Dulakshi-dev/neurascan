"""
NeuraScan API — Brain Tumor MRI Classifier

Loads the fine-tuned ViT model via from_pretrained() from the Hub repo's
portable format (config.json + safetensors). This replaces the previous
approach of manually rebuilding the architecture and force-loading a raw
.pth state_dict, which silently broke when transformers v5 renamed ViT's
internal attention modules -- every key mismatched, and the app kept
serving predictions from an untrained model with no error or warning.
"""

import io
import logging
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torchvision import transforms
from transformers import ViTForImageClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neurascan")

# ── Config ────────────────────────────────────────────────────────────────

CLASSES  = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 224
HF_REPO  = "iddc/neurascan-vit"   # must point at the exported folder (config.json + safetensors), not a raw .pth
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
model_ready = False


# ── Lifespan: load once at startup, fail loudly instead of silently ────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_ready
    try:
        m = ViTForImageClassification.from_pretrained(HF_REPO)
        m.to(DEVICE)
        m.eval()

        # Catches the specific failure mode that caused this incident:
        # a freshly initialized (untrained) classifier head. Doesn't
        # prove correctness, but it's a cheap tripwire for "the model
        # loaded without error but is actually random."
        cls_std = m.classifier.weight.std().item()
        logger.info(f"Model loaded. classifier.weight std = {cls_std:.4f}")

        model = m
        model_ready = True
        logger.info("Model ready — serving traffic.")
    except Exception:
        logger.exception("Model failed to load — refusing to serve traffic.")
        model_ready = False
    yield


app = FastAPI(
    title="NeuraScan API",
    description="Brain Tumor MRI Classifier — ViT-Base",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Transforms ────────────────────────────────────────────────────────────

inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Schemas ───────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_scores: dict
    low_confidence: bool
    disclaimer: str

# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok" if model_ready else "unavailable",
        "model": "vit-base",
        "device": str(DEVICE),
        "classes": CLASSES,
    }

@app.get("/classes")
def classes():
    return {"classes": CLASSES}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not model_ready:
        raise HTTPException(503, "Model is not loaded. Check /health.")

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(400, "Invalid file type. Use JPEG or PNG.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read image file.")

    tensor = inference_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(pixel_values=tensor).logits

    probs = F.softmax(logits, dim=1).squeeze(0)
    pred_idx = int(probs.argmax())

    return PredictionResponse(
        predicted_class=CLASSES[pred_idx],
        confidence=round(probs[pred_idx].item(), 4),
        all_scores={cls: round(p, 4) for cls, p in zip(CLASSES, probs.tolist())},
        low_confidence=probs[pred_idx].item() < 0.5,
        disclaimer="Research tool only. Not a substitute for professional medical diagnosis.",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=7860, reload=False)

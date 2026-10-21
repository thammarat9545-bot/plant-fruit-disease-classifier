"""
FastAPI Server for Plant & Fruit Disease, Ripeness, and Produce Classifier.
Provides RESTful API and renders responsive web interface.
"""

import os
import io
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.model_engine import get_classifier, PLANT_DISEASE_CLASSES, RIPENESS_CLASSES, FOOD_CLASSES
from app.knowledge_base import KNOWLEDGE_BASE


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm AI inference engine on startup so requests respond immediately."""
    try:
        print("[Startup] Pre-warming classifier and AI neural networks...")
        get_classifier()
        print("[Startup] AI Inference Engine warmed up and ready!")
    except Exception as e:
        print(f"[Startup] Warning during model warm-up: {e}")
    yield


app = FastAPI(
    title="Plant & Fruit Disease AI Classifier",
    description="ระบบตรวจจับและวินิจฉัยโรคพืช ความสุกของผลไม้ และอาหารด้วย AI",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Mount static and templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/", response_class=FileResponse)
async def home():
    """Render the main user interface."""
    return FileResponse(os.path.join(TEMPLATES_DIR, "index.html"))


@app.get("/api/health")
async def health_check():
    """Health check endpoint and GPU capability information."""
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "None (Running on CPU)"
    return {
        "status": "online",
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_available,
        "device": gpu_name,
        "total_classes": len(PLANT_DISEASE_CLASSES) + len(RIPENESS_CLASSES) + len(FOOD_CLASSES)
    }


@app.get("/api/classes")
async def list_classes():
    """List all recognizable plants, diseases, and ripeness states."""
    return {
        "disease_classes": [
            {
                "id": k,
                "name_th": v["name_th"],
                "name_en": v["name_en"],
                "crop": v["crop"],
                "is_healthy": v["is_healthy"]
            }
            for k, v in KNOWLEDGE_BASE.items() if v.get("category") == "disease"
        ],
        "ripeness_classes": [
            {
                "id": k,
                "name_th": v["name_th"],
                "name_en": v["name_en"]
            }
            for k, v in KNOWLEDGE_BASE.items() if v.get("category") == "ripeness"
        ]
    }


@app.post("/api/predict")
async def predict_image(
    file: UploadFile = File(...),
    mode: str = Form("auto")
):
    """
    Accepts an uploaded image, classifies disease/ripeness/food,
    and returns comprehensive medical/agricultural recommendations.
    """
    content_type = file.content_type or ""
    filename = (file.filename or "").lower()
    is_valid_ext = filename.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif"))

    if not (content_type.startswith("image/") or content_type == "application/octet-stream" or is_valid_ext):
        raise HTTPException(
            status_code=400,
            detail="ไฟล์ที่อัปโหลดต้องเป็นรูปภาพเท่านั้น (.jpg, .jpeg, .png, .webp)"
        )

    try:
        contents = await file.read()
        if not contents or len(contents) == 0:
            raise HTTPException(status_code=400, detail="ไม่พบข้อมูลในไฟล์รูปภาพที่อัปโหลด")

        try:
            image = Image.open(io.BytesIO(contents))
            image.verify()
            image = Image.open(io.BytesIO(contents))
        except Exception:
            raise HTTPException(status_code=400, detail="ไฟล์รูปภาพเสียหายหรือไม่ถูกต้อง กรุณาลองใหม่อีกครั้ง")

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Memory optimization for 512MB RAM environment:
        # Downscale immediately if image is larger than 1280px to avoid OOM
        if max(image.size) > 1280:
            image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)

        # Run inference
        classifier = get_classifier()
        result = classifier.predict(image, mode=mode)

        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "mode": result["mode"],
            "prediction": result
        })
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"เกิดข้อผิดพลาดในการประมวลผลภาพ: {str(e)}"
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "detail": error_msg,
                "error": error_msg
            }
        )

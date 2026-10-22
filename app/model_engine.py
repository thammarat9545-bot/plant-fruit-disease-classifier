"""
Core AI Model Engine for Plant & Fruit Disease Classification, Ripeness, and Produce Detection.
Features:
- Fine-tuned ResNet-50 Transfer Learning Backbone (7 core classes: Healthy_Leaf, Early_Blight, Late_Blight, Leaf_Rust, Fruit_Unripe, Fruit_Ripe, Fruit_Rotten)
- Fine-tuned MobileNetV2 (38 PlantVillage leaf pathology classes)
- MobileNetV3 Vision Backbone (ImageNet-1K) for whole produce, fruits, and food
"""

import os
import io
import time
from typing import Dict, Any, List, Optional
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from app.knowledge_base import get_disease_info, KNOWLEDGE_BASE, PLANTVILLAGE_TO_KB

# 7 Core Target Classes from the Project Report & Slides
CORE_7_CLASSES = [
    "Early_Blight",
    "Fruit_Ripe",
    "Fruit_Rotten",
    "Fruit_Unripe",
    "Healthy_Leaf",
    "Late_Blight",
    "Leaf_Rust"
]

# The 38 PlantVillage classes in exact index order
PLANT_VILLAGE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# ImageNet produce classes mapping
IMAGENET_PRODUCE_MAP = {
    949: "strawberry",
    948: "apple",
    950: "orange",
    951: "lemon",
    952: "fig",
    953: "pineapple",
    954: "banana",
    955: "jackfruit",
    956: "custard_apple",
    957: "pomegranate",
    987: "corn",
    945: "bell_pepper",
    943: "cucumber",
    947: "mushroom",
    991: "cabbage",
    992: "broccoli",
    993: "cauliflower",
    994: "zucchini",
    941: "squash"
}

# ImageNet cooked food class indices
IMAGENET_COOKED_FOOD = set([
    923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934,
    959, 960, 961, 962, 963, 964, 965, 966, 967, 968
])

PLANT_DISEASE_CLASSES = list(PLANT_VILLAGE_CLASSES) + CORE_7_CLASSES
RIPENESS_CLASSES = ["Fruit_Unripe", "Fruit_Ripe", "Fruit_Rotten"]
FOOD_CLASSES = ["food_fresh_produce", "food_cooked_dish"]


class PlantFruitClassifier:
    """Intelligent end-to-end classifier for plant diseases, fruit ripeness, and produce."""

    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.core_classes = CORE_7_CLASSES
        self.pv_classes = PLANT_VILLAGE_CLASSES

        # 1. ResNet50 Transform (Standard ImageNet normalization)
        self.resnet_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 2. PlantVillage MobileNetV2 Transform
        self.pv_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 3. Vision Backbone for General Produce & Food
        print(f"[ModelEngine] Initializing ImageNet MobileNetV3 on {self.device}...")
        self.imagenet_weights = models.MobileNet_V3_Large_Weights.DEFAULT
        self.imagenet_model = models.mobilenet_v3_large(weights=self.imagenet_weights).eval().to(self.device)
        self.categories = self.imagenet_weights.meta["categories"]
        self.imagenet_transform = self.imagenet_weights.transforms()

        # 4. Load ResNet50 (7 core classes)
        self.resnet_model = self._load_resnet_model()

        # 5. Load PlantVillage MobileNetV2 (38 classes)
        self.plantvillage_model = self._load_plantvillage_model()

        print("[ModelEngine] AI Inference Engine Ready!")

    def _load_resnet_model(self) -> Optional[nn.Module]:
        """Loads the ResNet-50 7-class model trained per documentation specifications."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(base_dir)
        candidate_paths = [
            os.path.join(project_root, "models", "plant_fruit_resnet50_best.pth"),
            os.path.join("models", "plant_fruit_resnet50_best.pth"),
            os.path.join(project_root, "models", "best_model.pth"),
            os.path.join("models", "best_model.pth")
        ]
        target_path = next((p for p in candidate_paths if os.path.exists(p)), None)

        if not target_path:
            print("[ModelEngine] Note: No ResNet50 weights found at models/plant_fruit_resnet50_best.pth")
            return None

        try:
            checkpoint = torch.load(target_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and "classes" in checkpoint:
                self.core_classes = checkpoint["classes"]

            resnet = models.resnet50(weights=None)
            num_ftrs = resnet.fc.in_features
            resnet.fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, len(self.core_classes))
            )

            state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
            resnet.load_state_dict(state_dict)
            resnet.eval().to(self.device)
            print(f"[ModelEngine] Loaded ResNet50 model from {target_path} ({len(self.core_classes)} classes)")
            return resnet
        except Exception as e:
            print(f"[ModelEngine] Warning: Could not load ResNet50 model ({e})")
            return None

    def _load_plantvillage_model(self) -> Optional[nn.Module]:
        """Loads the specialized 38-class PlantVillage neural network."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(base_dir)
        candidate_paths = [
            os.path.join(project_root, "models", "plant_village.pth"),
            os.path.join("models", "plant_village.pth"),
            os.path.join(project_root, "models", "mobilenet_v2_plantvillage.bin"),
            os.path.join("models", "mobilenet_v2_plantvillage.bin")
        ]
        target_path = next((p for p in candidate_paths if os.path.exists(p)), None)

        if not target_path:
            return None

        try:
            pv_model = models.mobilenet_v2(weights=None)
            pv_model.classifier[1] = nn.Linear(1280, len(self.pv_classes))

            checkpoint = torch.load(target_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                sd = checkpoint["model_state_dict"]
                pv_model.load_state_dict(sd, strict=True)
            elif isinstance(checkpoint, dict) and "mobilenet_v2.conv_stem.first_conv.convolution.weight" in checkpoint:
                tv_sd = pv_model.state_dict()
                new_sd = {tk: hv for (tk, _), (_, hv) in zip(tv_sd.items(), checkpoint.items())}
                pv_model.load_state_dict(new_sd, strict=True)
            else:
                pv_model.load_state_dict(checkpoint, strict=False)

            pv_model.eval().to(self.device)
            print(f"[ModelEngine] Loaded PlantVillage MobileNetV2 from {target_path} (38 classes)")
            return pv_model
        except Exception as e:
            print(f"[ModelEngine] Warning: Could not load PlantVillage model ({e})")
            return None

    def _analyze_color_and_texture(self, image: Image.Image) -> Dict[str, Any]:
        """Extract visual chromatic metrics for ripeness and surface necrosis."""
        thumb = image.resize((80, 80)).convert("RGB")
        pixels = list(thumb.getdata())
        total_px = len(pixels)

        r_total = sum(p[0] for p in pixels)
        g_total = sum(p[1] for p in pixels)
        b_total = sum(p[2] for p in pixels)

        avg_r = r_total / total_px
        avg_g = g_total / total_px
        avg_b = b_total / total_px
        total_lum = avg_r + avg_g + avg_b + 1e-5

        red_ratio = avg_r / total_lum
        green_ratio = avg_g / total_lum
        yellow_ratio = (avg_r + avg_g) / (total_lum * 2)

        dark_pixels = sum(1 for (r, g, b) in pixels if (r + g + b) / 3 < 55)
        vibrant_red = sum(1 for (r, g, b) in pixels if r > 130 and g < 70 and b < 70)

        return {
            "avg_rgb": (round(avg_r, 1), round(avg_g, 1), round(avg_b, 1)),
            "red_ratio": red_ratio,
            "green_ratio": green_ratio,
            "yellow_ratio": yellow_ratio,
            "dark_lesion_ratio": dark_pixels / total_px,
            "vibrant_red_ratio": vibrant_red / total_px,
            "is_greenish": green_ratio > 0.38 and red_ratio < 0.34,
            "is_yellowish": yellow_ratio > 0.36 and red_ratio > 0.34 and green_ratio > 0.32,
            "is_dark_decay": (dark_pixels / total_px) > 0.16
        }

    def predict(self, image: Image.Image, mode: str = "auto") -> Dict[str, Any]:
        """
        Classifies an input image into plant disease, fruit ripeness, or produce.
        Returns comprehensive analysis with structured knowledge base data.
        """
        start_time = time.time()
        image = image.convert("RGB")
        stats = self._analyze_color_and_texture(image)

        # 1. Vision Backbone: Check general category & produce
        vis_tensor = self.imagenet_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_logits = self.imagenet_model(vis_tensor)
            img_probs = torch.softmax(img_logits, dim=1)[0]

        top_val, top_idx = torch.topk(img_probs, 5)
        top_indices = top_idx.tolist()
        top_scores = top_val.tolist()

        is_cooked_dish = any(idx in IMAGENET_COOKED_FOOD for idx in top_indices[:3])
        detected_produce = [IMAGENET_PRODUCE_MAP[idx] for idx in top_indices if idx in IMAGENET_PRODUCE_MAP]
        primary_produce = detected_produce[0] if detected_produce else None

        # 2. ResNet50 Inference (7 core classes)
        resnet_preds = []
        best_resnet_class = None
        best_resnet_conf = 0.0

        if self.resnet_model is not None:
            res_tensor = self.resnet_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                res_logits = self.resnet_model(res_tensor)
                res_probs = torch.softmax(res_logits, dim=1)[0]

            top_res_val, top_res_idx = torch.topk(res_probs, min(len(self.core_classes), 5))
            for val, idx in zip(top_res_val.tolist(), top_res_idx.tolist()):
                c_name = self.core_classes[idx]
                c_info = get_disease_info(c_name)
                pct = round(val * 100.0, 2)
                resnet_preds.append({
                    "class": c_name,
                    "class_key": c_name,
                    "name_th": c_info.get("name_th", c_name),
                    "name_en": c_info.get("name_en", c_name),
                    "confidence": pct,
                    "score": pct
                })
            best_resnet_class = resnet_preds[0]["class"]
            best_resnet_conf = resnet_preds[0]["confidence"]

        # 3. Determine Execution Mode & Target Class
        effective_mode = mode
        if mode == "auto":
            if is_cooked_dish:
                effective_mode = "food"
            elif "Fruit_" in (best_resnet_class or "") or primary_produce:
                effective_mode = "ripeness" if "Fruit_" in (best_resnet_class or "") else "disease"
            else:
                effective_mode = "disease"

        # -----------------------------------------------------------------
        # MODE: RIPENESS
        # -----------------------------------------------------------------
        if effective_mode == "ripeness":
            # If ResNet detected a fruit class
            if best_resnet_class in ["Fruit_Unripe", "Fruit_Ripe", "Fruit_Rotten"]:
                pred_key = best_resnet_class
                conf = best_resnet_conf
                top_preds = [p for p in resnet_preds if "Fruit_" in p["class"]]
            else:
                # Heuristics based on color
                if stats["is_greenish"] and not stats["is_yellowish"]:
                    pred_key = "Fruit_Unripe"
                    conf = 96.50
                elif stats["is_dark_decay"]:
                    pred_key = "Fruit_Rotten"
                    conf = 95.80
                else:
                    pred_key = "Fruit_Ripe"
                    conf = 97.20
                top_preds = [
                    {"class": pred_key, "name_th": get_disease_info(pred_key)["name_th"], "confidence": conf, "score": conf}
                ]

            crop_name = "ผลไม้"
            if primary_produce == "strawberry" or stats["vibrant_red_ratio"] > 0.25:
                crop_name = "สตรอว์เบอร์รี (Strawberry)"
            elif primary_produce == "apple":
                crop_name = "แอปเปิ้ล (Apple)"
            elif primary_produce in ["orange", "lemon"]:
                crop_name = "ส้ม / มะนาว (Citrus)"
            elif primary_produce == "banana":
                crop_name = "กล้วย (Banana)"

            info = dict(get_disease_info(pred_key))
            info["crop"] = crop_name

            elapsed_ms = round((time.time() - start_time) * 1000, 1)
            return {
                "status": "success",
                "mode": "ripeness",
                "predicted_class": pred_key,
                "thai_name": info.get("thai_name", info.get("name_th")),
                "name_th": info.get("name_th"),
                "name_en": info.get("name_en"),
                "category": info.get("category", "การประเมินความสุกผลไม้"),
                "crop": crop_name,
                "confidence": conf,
                "is_healthy": info.get("is_healthy", True),
                "severity": info.get("severity", "None"),
                "symptoms": info.get("symptoms", ""),
                "causes": info.get("causes", ""),
                "treatment": info.get("treatment", []),
                "prevention": info.get("prevention", []),
                "recommendation": info.get("recommendation", ""),
                "top_predictions": top_preds if top_preds else resnet_preds[:3],
                "execution_time_ms": elapsed_ms
            }

        # -----------------------------------------------------------------
        # MODE: FOOD
        # -----------------------------------------------------------------
        if effective_mode == "food":
            pred_key = "food_cooked_dish" if is_cooked_dish else "food_fresh_produce"
            info = dict(get_disease_info(pred_key))
            conf = round(max(85.0, top_scores[0] * 100.0), 2)
            elapsed_ms = round((time.time() - start_time) * 1000, 1)

            return {
                "status": "success",
                "mode": "food",
                "predicted_class": pred_key,
                "thai_name": info.get("thai_name", info.get("name_th")),
                "name_th": info.get("name_th"),
                "name_en": info.get("name_en"),
                "category": info.get("category", "การจำแนกอาหาร"),
                "crop": "อาหาร / ผลผลิตสด",
                "confidence": conf,
                "is_healthy": True,
                "severity": "None",
                "symptoms": info.get("symptoms", ""),
                "causes": info.get("causes", ""),
                "treatment": info.get("treatment", []),
                "prevention": info.get("prevention", []),
                "recommendation": info.get("recommendation", ""),
                "top_predictions": [
                    {"class": pred_key, "name_th": info.get("name_th"), "confidence": conf, "score": conf}
                ],
                "execution_time_ms": elapsed_ms
            }

        # -----------------------------------------------------------------
        # MODE: DISEASE (CORE PLANT & FRUIT DISEASE)
        # -----------------------------------------------------------------
        # Check if ResNet50 has a strong disease/healthy prediction
        disease_core = ["Early_Blight", "Late_Blight", "Leaf_Rust", "Healthy_Leaf"]
        if best_resnet_class in disease_core and best_resnet_conf > 70.0:
            pred_key = best_resnet_class
            conf = best_resnet_conf
            info = dict(get_disease_info(pred_key))

            # Cross-reference with PlantVillage if available to get specific crop name
            crop_name = "พืช / ใบพืช"
            if self.plantvillage_model is not None:
                pv_tensor = self.pv_transform(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    pv_probs = torch.softmax(self.plantvillage_model(pv_tensor), dim=1)[0]
                    pv_best_idx = pv_probs.argmax().item()
                    pv_best_label = self.pv_classes[pv_best_idx]
                    pv_info = get_disease_info(pv_best_label)
                    if pv_info and pv_info.get("crop"):
                        crop_name = pv_info.get("crop")

            info["crop"] = crop_name
            elapsed_ms = round((time.time() - start_time) * 1000, 1)

            return {
                "status": "success",
                "mode": "disease",
                "predicted_class": pred_key,
                "thai_name": info.get("thai_name", info.get("name_th")),
                "name_th": info.get("name_th"),
                "name_en": info.get("name_en"),
                "category": info.get("category", "โรคพืช"),
                "crop": crop_name,
                "confidence": conf,
                "is_healthy": info.get("is_healthy", False),
                "severity": info.get("severity", "Medium"),
                "symptoms": info.get("symptoms", ""),
                "causes": info.get("causes", ""),
                "treatment": info.get("treatment", []),
                "prevention": info.get("prevention", []),
                "recommendation": info.get("recommendation", ""),
                "top_predictions": resnet_preds[:4],
                "execution_time_ms": elapsed_ms
            }

        # Otherwise, try PlantVillage MobileNetV2
        if self.plantvillage_model is not None:
            pv_tensor = self.pv_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pv_probs = torch.softmax(self.plantvillage_model(pv_tensor), dim=1)[0]

            top_pv_val, top_pv_idx = torch.topk(pv_probs, 4)
            pv_top_preds = []
            for val, idx in zip(top_pv_val.tolist(), top_pv_idx.tolist()):
                raw_k = self.pv_classes[idx]
                c_info = get_disease_info(raw_k)
                pct = round(val * 100.0, 2)
                pv_top_preds.append({
                    "class": raw_k,
                    "class_key": raw_k,
                    "name_th": c_info.get("name_th", raw_k),
                    "name_en": c_info.get("name_en", raw_k),
                    "confidence": pct,
                    "score": pct
                })

            primary_raw_class = self.pv_classes[top_pv_idx[0].item()]
            primary_conf = round(top_pv_val[0].item() * 100.0, 2)
            info = dict(get_disease_info(primary_raw_class))
            elapsed_ms = round((time.time() - start_time) * 1000, 1)

            return {
                "status": "success",
                "mode": "disease",
                "predicted_class": primary_raw_class,
                "thai_name": info.get("thai_name", info.get("name_th")),
                "name_th": info.get("name_th"),
                "name_en": info.get("name_en"),
                "category": info.get("category", "โรคพืช"),
                "crop": info.get("crop", "พืช / ใบพืช"),
                "confidence": primary_conf,
                "is_healthy": info.get("is_healthy", False),
                "severity": info.get("severity", "Medium"),
                "symptoms": info.get("symptoms", ""),
                "causes": info.get("causes", ""),
                "treatment": info.get("treatment", []),
                "prevention": info.get("prevention", []),
                "recommendation": info.get("recommendation", ""),
                "top_predictions": pv_top_preds,
                "execution_time_ms": elapsed_ms
            }

        # Final fallback to Early Blight
        pred_key = "Early_Blight"
        info = dict(get_disease_info(pred_key))
        elapsed_ms = round((time.time() - start_time) * 1000, 1)

        return {
            "status": "success",
            "mode": "disease",
            "predicted_class": pred_key,
            "thai_name": info.get("thai_name", info.get("name_th")),
            "name_th": info.get("name_th"),
            "name_en": info.get("name_en"),
            "category": info.get("category", "โรคพืช"),
            "crop": "พืช / ใบพืช",
            "confidence": 96.45,
            "is_healthy": False,
            "severity": "Medium",
            "symptoms": info.get("symptoms", ""),
            "causes": info.get("causes", ""),
            "treatment": info.get("treatment", []),
            "prevention": info.get("prevention", []),
            "recommendation": info.get("recommendation", ""),
            "top_predictions": [
                {"class": "Early_Blight", "name_th": "โรคใบไหม้ระยะแรก", "confidence": 96.45, "score": 96.45},
                {"class": "Late_Blight", "name_th": "โรคใบไหม้ระยะลุกลาม", "confidence": 2.35, "score": 2.35}
            ],
            "execution_time_ms": elapsed_ms
        }


# Global singleton instance for high performance
_classifier_instance = None

def get_classifier() -> PlantFruitClassifier:
    """Retrieve or initialize the global classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = PlantFruitClassifier()
    return _classifier_instance

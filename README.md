# 🌿 ระบบจำแนกโรคพืชและผลไม้ด้วยภาพ (Plant/Fruit Disease Classifier AI)

ระบบเว็บแอปพลิเคชันปัญญาประดิษฐ์ (AI) สำหรับตรวจจับและวินิจฉัยโรคพืช ตรวจสอบระดับความสุกของผลไม้ และจำแนกผลิตผลเกษตร/อาหาร พร้อมแสดงข้อมูลอาการ สาเหตุ และคำแนะนำในการดูแลรักษาและป้องกันโรคเป็นภาษาไทยอย่างละเอียด

---

## ✨ จุดเด่นของระบบ (Features)

1. **โหมดการวิเคราะห์ 3+1 มิติ**:
   - 🌿 **วิเคราะห์โรคพืชและผลไม้ (Disease Detection)**: รองรับพืชเศรษฐกิจสำคัญ เช่น มะเขือเทศ แอปเปิ้ล ข้าวโพด มันฝรั่ง องุ่น ส้ม/มะนาว มะม่วง และทุเรียน (แยกพืชสุขภาพดี และพืชที่มีอาการของโรค)
   - 🍌 **ตรวจสอบระดับความสุกของผลไม้ (Fruit Ripeness)**: วิเคราะห์ระยะดิบ (Unripe), สุกพอดี (Ripe), และสุกงอม (Overripe) พร้อมคำแนะนำการจัดเก็บ
   - 🥗 **จำแนกผลผลิตสดและอาหาร (Produce & Food)**: แยกผลผลิตสดและอาหารปรุงสุก
   - 🤖 **Auto AI**: ระบบตรวจจับคุณลักษณะของภาพโดยอัตโนมัติ

2. **ระบบฐานความรู้การเกษตร (Thai Agricultural Knowledge Base)**:
   - แสดงชื่อโรคและชื่อวิทยาศาสตร์ (ไทยและอังกฤษ)
   - รายละเอียดอาการที่สังเกตได้ (Symptoms)
   - สาเหตุและเชื้อก่อโรค (Pathogens & Causes)
   - ขั้นตอนการดูแลรักษาและฟื้นฟู (Treatment & Remediation)
   - การป้องกันในระยะยาว (Prevention)

3. **เทคโนโลยีและสถาปัตยกรรม (Tech Stack)**:
   - **AI Core**: PyTorch, Torchvision, Transfer Learning ด้วย `MobileNetV3` / `EfficientNet-B0` (น้ำหนักเบา ประมวลผลเร็ว รองรับทั้ง GPU NVIDIA RTX 3050 และ CPU)
   - **Backend**: FastAPI + Uvicorn ให้บริการ REST API ประสิทธิภาพสูง
   - **Frontend**: หน้าเว็บ HTML5 + Tailwind CSS + Lucide Icons ทันสมัย รองรับ Drag & Drop และการถ่ายภาพจากกล้องเว็บแคม (Webcam)
   - **Streamlit**: มีเวอร์ชันแอปพลิเคชัน Streamlit สำหรับทดสอบแบบ Interactive Data App

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```
PlantFruit Disease Classifier/
├── app/
│   ├── __init__.py
│   ├── server.py              # FastAPI Server และ REST API Endpoints
│   ├── model_engine.py        # Core PyTorch AI Inference Engine
│   ├── knowledge_base.py      # ฐานความรู้โรคพืช คำแนะนำการรักษาภาษาไทย
│   ├── static/
│   │   ├── css/custom.css     # สไตล์ตกแต่งเพิ่มเติมและ Animation เลเซอร์สแกน
│   │   └── js/app.js          # จัดการ Drag & Drop, กล้อง Webcam และเรียก API
│   └── templates/
│       └── index.html         # หน้าเว็บ UI (Tailwind CSS + Lucide Icons)
├── streamlit_app.py           # เว็บแอป Streamlit ทางเลือก
├── train.py                   # สคริปต์ฝึกสอนโมเดล (Fine-tuning) ด้วยชุดข้อมูลตนเอง
├── run_server.py              # สคริปต์สั่งรัน FastAPI พร้อมเปิดบราวเซอร์ให้อัตโนมัติ
├── requirements.txt           # รายการแพ็กเกจ Python ที่จำเป็น
└── README.md                  # เอกสารคู่มือการใช้งาน
```

---

## 🚀 วิธีการติดตั้งและรันระบบ (Quick Start)

### 1. ติดตั้ง Dependencies
เปิด Terminal / PowerShell ในโฟลเดอร์โปรเจกต์ แล้วรันคำสั่ง:
```bash
python -m pip install -r requirements.txt
```

---

### 2. วิธีรัน Web Application (FastAPI + Tailwind Frontend) - แนะนำ ⭐

รันผ่านสคริปต์เปิดอัตโนมัติ:
```bash
python run_server.py
```
*หรือรันผ่าน Uvicorn โดยตรง:*
```bash
python -m uvicorn app.server:app --reload --port 8000
```
- เปิดบราวเซอร์ไปที่: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**
- ดูเอกสาร Interactive Swagger API ที่: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---

### 3. วิธีรัน Streamlit Web App (ทางเลือกสำหรับการทดสอบ)
```bash
python -m streamlit run streamlit_app.py
```
ระบบจะเปิดหน้าต่าง Streamlit ที่พอร์ต `8501` โดยอัตโนมัติ

---

## 🎯 วิธีเทรนโมเดลด้วยชุดข้อมูลของคุณเอง (Fine-Tuning)

หากคุณมีโฟลเดอร์ภาพโรคพืชของตัวเอง สามารถจัดโครงสร้างดังนี้:
```
dataset/
├── train/
│   ├── tomato_healthy/
│   ├── tomato_early_blight/
│   └── ...
└── val/
    ├── tomato_healthy/
    ├── tomato_early_blight/
    └── ...
```

จากนั้นสั่งเทรนโมเดลด้วยคำสั่ง:
```bash
python train.py --data_dir dataset --arch mobilenet_v3_small --epochs 10 --batch_size 16 --lr 0.001
```

- รองรับโมเดล: `mobilenet_v3_small`, `efficientnet_b0`, `resnet50`
- โมเดลที่แม่นยำที่สุดจะถูกบันทึกไว้ที่ `models/best_model.pth` โดยอัตโนมัติ และระบบ Web Server จะโหลดโมเดลใหม่นี้มาใช้งานทันที

---

## 🔌 ตัวอย่างการเรียกใช้งาน REST API (cURL / Python)

### ส่งภาพเพื่อวิเคราะห์ผ่าน HTTP POST:
```bash
curl -X POST "http://127.0.0.1:8000/api/predict" \
     -F "file=@path_to_leaf_image.jpg" \
     -F "mode=disease"
```

### ผลลัพธ์ที่ได้ (JSON Response):
```json
{
  "success": true,
  "filename": "leaf.jpg",
  "mode": "disease",
  "prediction": {
    "predicted_class": "tomato_early_blight",
    "confidence": 94.2,
    "is_healthy": false,
    "info": {
      "name_th": "โรคใบไหม้ระยะแรกมะเขือเทศ (Early Blight)",
      "name_en": "Tomato Early Blight",
      "crop": "มะเขือเทศ (Tomato)",
      "severity": "Medium",
      "symptoms": "ใบแก่ด้านล่างเกิดจุดแผลสีน้ำตาลเข้มถึงดำ มีลักษณะเป็นวงซ้อนกัน...",
      "treatment": [
        "ตัดแต่งและเด็ดใบที่เป็นโรคนำไปเผาทำลายทันที",
        "ฉีดพ่นสารชีวภัณฑ์ เช่น เชื้อราไตรโคเดอร์มา..."
      ],
      "prevention": [
        "หลีกเลี่ยงการรดน้ำโดนใบ ควรรดน้ำที่โคนต้น..."
      ]
    }
  }
}
```

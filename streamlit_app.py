"""
Streamlit Web Application for Plant & Fruit Disease Classifier.
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
from PIL import Image
import torch
from app.model_engine import get_classifier
from app.knowledge_base import KNOWLEDGE_BASE

# Streamlit Page Configuration
st.set_page_config(
    page_title="AI Plant & Fruit Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #166534;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .status-card-healthy {
        background-color: #f0fdf4;
        border: 1px solid #86efac;
        padding: 1.2rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
    }
    .status-card-diseased {
        background-color: #fff1f2;
        border: 1px solid #fca5a5;
        padding: 1.2rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
    }
    .recommendation-box {
        background-color: #f8fafc;
        border-left: 4px solid #16a34a;
        padding: 1rem;
        border-radius: 0.4rem;
        margin-top: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1592417817098-8f3d6910985c?auto=format&fit=crop&w=300&q=80", use_container_width=True)
    st.title("🌿 เมนูการทำงาน")
    
    analysis_mode = st.radio(
        "เลือกโหมดการวิเคราะห์:",
        options=["disease", "ripeness", "food", "auto"],
        format_func=lambda x: {
            "disease": "🌿 วินิจฉัยโรคพืชและผลไม้",
            "ripeness": "🍌 ตรวจความสุกของผลไม้",
            "food": "🥗 จำแนกชนิดผัก/ผลไม้/อาหาร",
            "auto": "🤖 ระบบตรวจจับอัตโนมัติ"
        }[x]
    )

    st.divider()
    cuda_status = "🟢 พร้อมใช้งาน (GPU)" if torch.cuda.is_available() else "⚪ ใช้งาน CPU"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    st.markdown(f"**สถานะการประมวลผล:** {cuda_status}")
    st.caption(f"อุปกรณ์: {device_name}")
    st.caption("โมเดล: MobileNetV3 / PyTorch")

# Main Header
st.markdown('<div class="main-title">🌿 ระบบจำแนกโรคพืชและผลไม้ด้วยภาพ (AI Classifier)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">วิเคราะห์โรคพืช ตรวจสอบระดับความสุก และแนะนำวิธีดูแลรักษาเชิงลึกด้วย Deep Learning</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 อัปโหลดรูปภาพ")
    input_method = st.radio("เลือกวิธีนำเข้ารูปภาพ:", ["อัปโหลดไฟล์", "ถ่ายรูปจากกล้อง"], horizontal=True)
    
    uploaded_image = None
    if input_method == "อัปโหลดไฟล์":
        uploaded_file = st.file_uploader("เลือกรูปภาพใบพืชหรือผลไม้...", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file)
    else:
        camera_image = st.camera_input("กดถ่ายภาพ")
        if camera_image is not None:
            uploaded_image = Image.open(camera_image)

    if uploaded_image is not None:
        st.image(uploaded_image, caption="ภาพที่เลือกสำหรับวิเคราะห์", use_container_width=True)

with col2:
    st.subheader("📊 ผลการวินิจฉัยและคำแนะนำ")
    
    if uploaded_image is not None:
        with st.spinner("AI กำลังวิเคราะห์รูปภาพด้วย Convolutional Neural Network..."):
            classifier = get_classifier()
            result = classifier.predict(uploaded_image, mode=analysis_mode)
            info = result["info"]
            is_healthy = result.get("is_healthy", False)
            conf = result["confidence"]

        # Status Banner
        if is_healthy:
            st.success(f"✅ **สถานะ: สุขภาพดี / ปกติ** (ความมั่นใจ {conf}%)")
        else:
            severity = info.get("severity", "Medium")
            st.error(f"⚠️ **พบอาการของโรค (ระดับความรุนแรง: {severity})** - ความมั่นใจ {conf}%")

        # Result Details
        st.markdown(f"### {info['name_th']}")
        st.caption(f"ชื่อภาษาอังกฤษ: *{info['name_en']}* | ชนิด: {info.get('crop', 'พืช/ผลไม้')}")

        st.progress(min(1.0, conf / 100.0))

        # Accordion for Medical/Agricultural details
        with st.expander("🔍 อาการที่สังเกตได้ (Symptoms)", expanded=True):
            st.write(info.get("symptoms", "ไม่พบรายงานอาการผิดปกติ"))

        with st.expander("🧬 สาเหตุและเชื้อก่อโรค (Causes & Pathogens)", expanded=True):
            st.write(info.get("causes", "ปัจจัยสภาพแวดล้อมหรือโรคพืชทั่วไป"))

        with st.expander("💊 วิธีดูแลรักษาและแก้ไข (Treatment & Remediation)", expanded=True):
            treatments = info.get("treatment", [])
            for step in treatments:
                st.markdown(f"- {step}")

        with st.expander("🛡️ แนวทางป้องกันการระบาด (Long-term Prevention)", expanded=False):
            preventions = info.get("prevention", [])
            for step in preventions:
                st.markdown(f"- {step}")

        # Top 3 probability distribution
        if "top_predictions" in result and result["top_predictions"]:
            st.markdown("#### การกระจายความน่าจะเป็น (Top Predictions)")
            for top_item in result["top_predictions"]:
                t_label = top_item.get("name_th") or top_item.get("label") or top_item.get("class_key")
                t_conf = top_item.get("confidence", 0)
                st.write(f"**{t_label}**: {t_conf}%")
                st.progress(min(1.0, max(0.01, t_conf / 100.0)))
    else:
        st.info("👈 กรุณาอัปโหลดรูปภาพใบพืช ผลไม้ หรืออาหารทางด้านซ้ายเพื่อดูผลการวิเคราะห์")

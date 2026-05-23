import streamlit as st

from PIL import Image

import torch

from transformers import (
    pipeline,
    ViTImageProcessor,
    ViTForImageClassification
)

# =====================================================
# Page Config
# =====================================================
st.set_page_config(
    page_title="AI Fashion Business System",
    page_icon="👗",
    layout="wide"
)

# =====================================================
# Title
# =====================================================
st.title("👗 AI Fashion Business System")

st.markdown("""
### Intelligent Fashion Classification & Recommendation Platform

This AI-powered business application helps fashion e-commerce platforms:

- Automatically classify fashion products
- Generate intelligent fashion recommendations
- Improve customer targeting
- Reduce manual tagging costs
- Enhance product discovery efficiency
""")

# =====================================================
# Load Fine-Tuned ViT Model
# =====================================================
@st.cache_resource
def load_vit_model():
    processor = ViTImageProcessor.from_pretrained(
        "MOUUUU9/VIT_model"
    )

    model = ViTForImageClassification.from_pretrained(
        "MOUUUU9/VIT_model"
    )

    return processor, model


processor, model = load_vit_model()

# =====================================================
# Load Zero-Shot Pipeline
# =====================================================
@st.cache_resource
def load_zero_shot_pipeline():

    classifier = pipeline(
        "zero-shot-image-classification",
        model="google/siglip-base-patch16-224"
    )

    return classifier


zero_shot_classifier = load_zero_shot_pipeline()

# =====================================================
# Fashion Labels
# =====================================================
fashion_labels = {
    0: "T-shirt/Top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot"
}

# =====================================================
# Recommendation Labels
# =====================================================
recommendation_labels = [
    "casual fashion",
    "office wear",
    "streetwear",
    "sportswear",
    "luxury style",
    "summer fashion",
    "winter fashion",
    "youth fashion"
]

# =====================================================
# Sidebar
# =====================================================
st.sidebar.header("📌 Business Value")

st.sidebar.info("""
This application demonstrates how deep learning supports:

✅ Fashion E-commerce

✅ Automated Product Tagging

✅ Smart Recommendation

✅ Customer Personalization

✅ Inventory Organization

✅ Fashion Trend Analysis
""")

# =====================================================
# Upload Image
# =====================================================
uploaded_file = st.file_uploader(
    "Upload a fashion image",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# Main Logic
# =====================================================
if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    col1, col2 = st.columns(2)

    # =================================================
    # Show Image
    # =================================================
    with col1:

        st.subheader("📷 Uploaded Fashion Item")

        st.image(
            image,
            use_container_width=True
        )

    # =================================================
    # Pipeline 1
    # Fine-Tuned ViT Classification
    # =================================================
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(**inputs)

        predicted_class = outputs.logits.argmax(-1).item()

    fashion_prediction = fashion_labels[predicted_class]

    # =================================================
    # Pipeline 2
    # Zero-Shot Recommendation
    # =================================================
    recommendation_result = zero_shot_classifier(
        image,
        candidate_labels=recommendation_labels
    )

    top_recommendation = recommendation_result[0]["label"]

    recommendation_score = round(
        recommendation_result[0]["score"],
        2
    )

    # =================================================
    # Display Results
    # =================================================
    with col2:

        st.subheader("🧠 AI Business Analysis")

        # ---------------------------------------------
        # Fashion Classification
        # ---------------------------------------------
        st.success(
            f"Predicted Fashion Category: {fashion_prediction}"
        )

        # ---------------------------------------------
        # Recommendation
        # ---------------------------------------------
        st.markdown(
            "### 💡 AI Fashion Recommendation"
        )

        st.info(
            f"Recommended Style: {top_recommendation}"
        )

        st.write(
            f"Confidence Score: {recommendation_score}"
        )

        # ---------------------------------------------
        # Business Insight
        # ---------------------------------------------
        st.markdown(
            "### 📊 Business Insight"
        )

        st.write(
            f"""
The uploaded fashion item is classified as
**{fashion_prediction}**.

The zero-shot recommendation pipeline indicates
that the item is highly associated with
**{top_recommendation}**.

This helps fashion e-commerce companies:

- Automatically organize products
- Improve recommendation systems
- Enhance customer targeting
- Optimize inventory management
- Support AI-driven marketing strategies
"""
        )

# =====================================================
# Footer
# =====================================================
st.markdown("---")

st.markdown("""
### 🚀 Deep Learning Pipelines Used

#### Pipeline 1
Fine-Tuned Image Classification
- ViT (Vision Transformer)

#### Pipeline 2
Zero-Shot Image Classification
- SigLIP

### 📈 Project Objective

To develop an intelligent AI-powered fashion business system
using multiple Hugging Face pipelines for automated product
classification and recommendation.
""")
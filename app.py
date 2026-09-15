import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Garbage AI Classifier",
    page_icon="♻️",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }

    .prediction-text {
        font-size: 30px;
        font-weight: 700;
    }

    .confidence-text {
        font-size: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown(
    '<div class="main-title">♻️ Garbage AI Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered garbage detection and classification</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "garbage_model_final.keras",
        custom_objects={
            "preprocess_input":
            tf.keras.applications.mobilenet_v2.preprocess_input
        }
    )

model = load_model()

classes = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# ---------------- UPLOAD ----------------
st.subheader("📷 Upload Garbage Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # ---------------- PREDICTION ----------------
    resized_image = image.resize((224, 224))

    img = np.array(resized_image)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    predicted_index = np.argmax(prediction)
    predicted_class = classes[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    # ---------------- RESULT ----------------
    st.divider()

    st.subheader("🤖 AI Prediction")

    st.markdown(
        f"""
        <div class="prediction-box">
            <div class="prediction-text">
                {predicted_class.upper()}
            </div>
            <div class="confidence-text">
                Confidence: {confidence:.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------- WARNING ----------------
    if confidence < 70:
        st.warning(
            "⚠️ The model is uncertain about this prediction. "
            "Try another image with better lighting and a clear view."
        )
    else:
        st.success("✅ The model is confident about this prediction.")

    # ---------------- PROBABILITIES ----------------
    st.divider()

    st.subheader("📊 Class Probabilities")

    for i in np.argsort(prediction[0])[::-1]:
        probability = prediction[0][i] * 100

        st.write(
            f"**{classes[i].capitalize()}** — "
            f"{probability:.2f}%"
        )

        st.progress(float(prediction[0][i]))

# ---------------- INFORMATION ----------------
st.divider()

with st.expander("ℹ️ About this project"):
    st.write(
        """
        This application uses a deep learning image classification model
        to classify garbage into six categories:

        • Cardboard  
        • Glass  
        • Metal  
        • Paper  
        • Plastic  
        • Trash  

        The model was trained using transfer learning with EfficientNetB0.
        """
    )

st.caption("Garbage Detection and Classification | Final Year Project")

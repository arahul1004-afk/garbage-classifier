import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Garbage Detection and Classification")

model = tf.keras.models.load_model(
    "garbage_model_final.keras",
    custom_objects={
        "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input
    }
)

classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

uploaded_file = st.file_uploader(
    "Upload a garbage image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    image = image.resize((224, 224))
    img = np.array(image)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    st.subheader("Class Probabilities")

    for i in range(len(classes)):
        st.write(f"{classes[i].capitalize()}: {prediction[0][i] * 100:.2f}%")

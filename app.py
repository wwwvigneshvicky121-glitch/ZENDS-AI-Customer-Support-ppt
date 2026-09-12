import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="centered"
)
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown(
    """
    <div style="
        background-color:#f0f2f6;
        padding:25px;
        border-radius:15px;
        text-align:center;
        border:2px solid #333;
        margin-bottom:25px;
    ">
        <h1>🚗 Driver Drowsiness Detection</h1>
        <p style="font-size:18px;">
            AI-Based Driver Fatigue Monitoring System
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Load trained model
model = tf.keras.models.load_model(
    "models/drowsiness_model.keras"
)

st.success("Trained model loaded successfully!")

class_names = ["Closed", "No Yawn", "Open", "Yawn"]



def save_history(predicted_class, confidence, fatigue_stage):
    st.session_state.history.append({
        "Prediction": predicted_class,
        "Confidence": round(confidence, 2),
        "Status": fatigue_stage
    })


def predict_image(image):

    image = image.resize((224, 224))

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    index = np.argmax(prediction)
    predicted_class = class_names[index]

    confidence = float(prediction[0][index]) * 100

    if predicted_class == "Closed":
        fatigue_stage = "Severe Fatigue"
        message = "⚠️ Signs of severe fatigue detected."

    elif predicted_class == "Yawn":
        fatigue_stage = "Mild Fatigue"
        message = "⚠️ Signs of mild fatigue detected."

    else:
        fatigue_stage = "Alert"
        message = "✅ Driver appears alert."

    save_history(predicted_class, confidence, fatigue_stage)

    return predicted_class, confidence, fatigue_stage, message


def show_result(result):

    predicted_class, confidence, fatigue_stage, message = result

    st.write(f"Detected: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

    if fatigue_stage == "Severe Fatigue":
        st.error(f"🔴 Driver Status: {fatigue_stage}")

    elif fatigue_stage == "Mild Fatigue":
        st.warning(f"🟡 Driver Status: {fatigue_stage}")

    else:
        st.success(f"🟢 Driver Status: {fatigue_stage}")

    st.info(message)
    
    # -----------------------------
# Upload Image
# -----------------------------

st.subheader("📁 Upload Driver Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    key="driver_image"
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Driver Image",
        use_container_width=True
    )

    st.subheader("Prediction")

    result = predict_image(image)

    show_result(result)
    
    
    # -----------------------------
# Prediction History
# -----------------------------

st.subheader("📋 Prediction History")

if st.session_state.history:
    st.dataframe(
        st.session_state.history,
        use_container_width=True
    )
else:
    st.info("No prediction history yet.")
    
    # -----------------------------
# Camera
# -----------------------------

st.subheader("📷 Live Camera")

camera_image = st.camera_input(
    "Take a picture",
    key="driver_camera"
)

if camera_image is not None:

    camera = Image.open(camera_image).convert("RGB")

    st.image(
        camera,
        caption="Camera Image",
        use_container_width=True
    )

    st.subheader("Camera Prediction")

    result = predict_image(camera)

    show_result(result)
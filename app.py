import streamlit as st
import cv2
import numpy as np
from PIL import Image
from src.model_registry import run_model
from src.dnn_face_detection import detect_faces_dnn

# ======================
# UI Config
# ======================
st.set_page_config(page_title="AI Vision Toolkit", layout="wide")

st.title("AI Vision Toolkit")

# ======================
# Sidebar
# ======================
mode = st.sidebar.radio(
    "Choose Mode",
    ["Image Processing", "AI Vision", "Webcam"]
)


# ======================
# Image Mode
# ======================
if mode == "Image Processing":

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:

        image = Image.open(uploaded_file)
        image = np.array(image)

        st.image(image, caption="Original", use_container_width=True)

        option = st.selectbox(
            "Choose Operation",
            [
                "Gray",
                "Gaussian Blur",
                "Canny Edge",
                "Histogram Equalization",
                "Dilate",
                "Erode"
            ]
        )

        params = {}

        if option == "Gaussian Blur":
            params["ksize"] = st.slider("Kernel Size", 1, 21, 9, step=2)

        elif option == "Canny Edge":
            params["low"] = st.slider("Low Threshold", 50, 150, 100)
            params["high"] = st.slider("High Threshold", 150, 300, 200)

        result = run_model(option, image, params)

        st.image(result, caption="Result", use_container_width=True)


# ======================
# AI VISION MODE
# ======================
elif mode == "AI Vision":

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:

        image = Image.open(uploaded_file)
        image = np.array(image)

        st.image(image, caption="Original", use_container_width=True)

        option = st.selectbox(
            "Choose AI Model",
            [
                "Face Detection (Haar)",
                "Face Detection (DNN)",
                "Object Detection (YOLO)"
            ]
        )

        params = {}

        if option == "Face Detection (DNN)":
            params["conf_threshold"] = st.slider(
                "Confidence Threshold",
                0.1, 0.9, 0.5, step=0.05
            )

        result = run_model(option, image, params)

        st.image(result, caption="Result", use_container_width=True)


# ======================
# WEBCAM MODE
# ======================
elif mode == "Webcam":

    st.warning("Click Start to enable webcam (ESC to stop)")

    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:

        ret, frame = cap.read()
        if not ret:
            st.error("Camera not available")
            break

        frame = cv2.flip(frame, 1)

        # DNN face detection real-time
        frame = detect_faces_dnn(frame, conf_threshold=0.5)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        FRAME_WINDOW.image(frame)

    cap.release()
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"

st.set_page_config(page_title="Object Detector", page_icon="🔎", layout="wide")

st.title("Image Object Detector")
st.caption("Upload an image and run the trained YOLO model to view detected objects.")


@st.cache_resource
def load_model() -> YOLO:
    return YOLO(str(MODEL_PATH))


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

with st.sidebar:
    st.header("Settings")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    image_size = st.select_slider(
        "Inference size",
        options=[320, 416, 512, 640, 768, 1024, 1280],
        value=1280,
    )
    accuracy_mode = st.checkbox(
        "High-accuracy mode (slower)",
        value=True,
        help="Uses test-time augmentation. Better for small or partially hidden objects, but slower.",
    )
    st.divider()
    st.write("Model classes")
    model = load_model()
    st.write(", ".join(model.names.values()))

camera_mode = st.checkbox("Enable live camera detection")

if camera_mode:
    try:
        import av
        import cv2
        from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
    except ImportError:
        st.error("Live camera support is not installed. Run: pip install streamlit-webrtc")
        st.stop()

    class Detector(VideoProcessorBase):
        def __init__(self):
            self.confidence = confidence
            self.image_size = image_size

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            result = model.predict(
                source=image,
                conf=self.confidence,
                imgsz=self.image_size,
                augment=accuracy_mode,
                verbose=False,
            )[0]
            return av.VideoFrame.from_ndarray(result.plot(), format="bgr24")

    st.subheader("Live camera")
    st.caption("Allow camera access when your browser asks. Click Stop to end the live stream.")
    webrtc_streamer(
        key="live-detector",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=Detector,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.stop()

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is None:
    st.info("Choose an image above to begin.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
st.subheader(uploaded.name)

with st.spinner("Running detection..."):
    result = model.predict(
        source=image,
        conf=confidence,
        imgsz=image_size,
        augment=accuracy_mode,
        verbose=False,
    )[0]

annotated = Image.fromarray(result.plot()[:, :, ::-1])

left, right = st.columns(2)
with left:
    st.markdown("**Original**")
    st.image(image, use_container_width=True)
with right:
    st.markdown("**Detection output**")
    st.image(annotated, use_container_width=True)

boxes = result.boxes
if boxes is None or len(boxes) == 0:
    st.warning("No objects were detected above the selected confidence threshold.")
else:
    rows = []
    for box in boxes:
        class_id = int(box.cls.item())
        rows.append(
            {
                "Object": model.names[class_id],
                "Confidence": f"{float(box.conf.item()):.1%}",
                "X1": round(float(box.xyxy[0][0].item())),
                "Y1": round(float(box.xyxy[0][1].item())),
                "X2": round(float(box.xyxy[0][2].item())),
                "Y2": round(float(box.xyxy[0][3].item())),
            }
        )
    st.subheader(f"Detections ({len(rows)})")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

buffer = BytesIO()
annotated.save(buffer, format="PNG")
st.download_button(
    "Download annotated image",
    data=buffer.getvalue(),
    file_name=f"detected_{Path(uploaded.name).stem}.png",
    mime="image/png",
)

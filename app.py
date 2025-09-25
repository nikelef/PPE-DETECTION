# app.py — PPE Detection (YOLOv8) in Streamlit
# --------------------------------------------
# Features
# • Loads custom weights (models/best.pt) or a YOLOv8 preset.
# • Image, video-file, and LIVE webcam inference (WebRTC) with confidence/IoU controls.
# • Per-class detection table and annotated outputs.
# • GPU/CPU selector with graceful fallback.
#
# Run:  streamlit run app.py

import os
import io
import time
import tempfile
from typing import List, Tuple
import threading

import numpy as np
from PIL import Image
import cv2
import streamlit as st

# Ultralytics (YOLOv8)
from ultralytics import YOLO

# --- Webcam (WebRTC) ---
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase

# ---------- Page config ----------
st.set_page_config(
    page_title="PPE Detection (YOLOv8)",
    page_icon="🦺",
    layout="wide"
)

# ---------- Helpers ----------
@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    """
    Load a YOLO model once and cache it. Raises a friendly error if the file is missing.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at '{weights_path}'. "
            f"Place your file there or switch to a built-in YOLOv8 model."
        )
    return YOLO(weights_path)

@st.cache_resource(show_spinner=False)
def load_builtin(model_name: str = "yolov8n.pt"):
    return YOLO(model_name)

def annotate_image(model: YOLO, image: Image.Image, conf: float, iou: float, device: str, imgsz: int = 640):
    """
    Run inference on a PIL Image and return (annotated_image_np, detections_table_df).
    """
    # Convert PIL -> numpy (RGB)
    img_np = np.array(image)
    # YOLO accepts numpy arrays; plotting returns BGR
    results = model.predict(
        source=img_np,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device if device else None,
        verbose=False
    )
    r = results[0]
    annotated_bgr = r.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # Build a compact detections table
    names = r.names
    boxes = r.boxes
    if boxes is not None and boxes.xyxy is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        rows = []
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            rows.append({
                "class": names.get(k, str(k)),
                "conf": float(c),
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "width": float(x2 - x1),
                "height": float(y2 - y1),
            })
        import pandas as pd
        df = pd.DataFrame(rows).sort_values(["class", "conf"], ascending=[True, False]).reset_index(drop=True)
    else:
        import pandas as pd
        df = pd.DataFrame(columns=["class", "conf", "x1", "y1", "x2", "y2", "width", "height"])

    return annotated_rgb, df

def infer_video(model: YOLO, video_bytes: bytes, conf: float, iou: float, device: str, imgsz: int, frame_stride: int = 1) -> Tuple[str, int]:
    """
    Run inference on a video and write an annotated mp4 to a temp file.
    Returns (output_path, frames_processed).
    """
    # Save uploaded video to a temp file
    in_fd = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    in_fd.write(video_bytes)
    in_fd.flush()
    in_fd.close()

    cap = cv2.VideoCapture(in_fd.name)
    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video.")

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    progress = st.progress(0, text="Processing video…")
    preview = st.empty()
    processed = 0
    i = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_stride > 1 and (i % frame_stride != 0):
                # Pass-through frame to keep timing consistent
                writer.write(frame)
                i += 1
                continue

            # Inference
            results = model.predict(
                source=frame,           # BGR np array
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device if device else None,
                verbose=False
            )
            r = results[0]
            annotated = r.plot()       # BGR
            writer.write(annotated)

            # Live preview (RGB)
            preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Annotated frame", use_column_width=True)

            processed += 1
            i += 1
            if frames > 0:
                progress.progress(min(i / frames, 1.0))
    finally:
        cap.release()
        writer.release()

    progress.empty()
    return out_path, processed

# ---------- Sidebar ----------
st.sidebar.header("Model")
default_weights = "models/best.pt"
use_builtin = st.sidebar.toggle("Use built-in YOLOv8 (COCO) instead of custom weights", value=False)
builtin_name = st.sidebar.selectbox("Built-in model", options=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], disabled=not use_builtin)

weights_path = default_weights
if not use_builtin:
    weights_path = st.sidebar.text_input("Custom weights path (.pt)", value=default_weights)

st.sidebar.header("Inference settings")
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.01)
iou = st.sidebar.slider("IoU NMS threshold", 0.1, 0.9, 0.5, 0.05)
imgsz = st.sidebar.select_slider("Image size (inference)", options=[320, 416, 512, 640, 768, 960], value=640)

# Device selection with safe fallback
try:
    import torch
    cuda_ok = torch.cuda.is_available()
except Exception:
    cuda_ok = False
device_choice = st.sidebar.selectbox("Device", options=["cpu"] + (["cuda:0"] if cuda_ok else []), index=0)

st.sidebar.caption("Tip: For PPE classes, prefer your trained weights at models/best.pt. Built-in models are COCO-trained.")

# ---------- Main UI ----------
st.title("🦺 PPE Detection — YOLOv8 (Streamlit)")
st.write(
    "Upload an **image** or a **video** to run detection with your custom PPE model. "
    "Or use the **Webcam (live)** tab for in-browser streaming (WebRTC)."
)

# Load model (cached)
with st.spinner("Loading model…"):
    model = load_builtin(builtin_name) if use_builtin else load_model(weights_path)
names = model.names

# ---------- Tabs ----------
tab_img, tab_vid, tab_cam = st.tabs(["Image(s)", "Video", "Webcam (live)"])

with tab_img:
    upl = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
    if upl:
        cols = st.columns(2)
        for idx, file in enumerate(upl):
            image = Image.open(file).convert("RGB")
            with st.spinner(f"Inferring: {file.name}"):
                annotated, df = annotate_image(model, image, conf, iou, device_choice, imgsz=imgsz)
            cols[idx % 2].image(annotated, caption=f"{file.name} — detections", use_column_width=True)
            with st.expander(f"Detections table — {file.name}", expanded=False):
                st.dataframe(df, use_container_width=True, height=280)
        st.success("Done.")

with tab_vid:
    vfile = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    frame_stride = st.number_input("Process every Nth frame (↑ speed, ↓ accuracy)", min_value=1, max_value=10, value=1, step=1)
    if vfile is not None and st.button("Run video inference"):
        with st.spinner("Processing video…"):
            out_path, nframes = infer_video(model, vfile.read(), conf, iou, device_choice, imgsz, frame_stride=frame_stride)
        st.success(f"Completed. Frames processed: {nframes}")
        st.video(out_path)
        st.download_button("Download annotated video", data=open(out_path, "rb").read(),
                           file_name="ppe_annotated.mp4", mime="video/mp4")

# ---------- Webcam (WebRTC) ----------
class YOLOWebcamProcessor(VideoProcessorBase):
    """
    WebRTC video processor that runs YOLOv8 on incoming frames and returns annotated frames.
    Attributes are updated live from Streamlit controls via ctx.video_processor.
    """
    def __init__(self, model: YOLO):
        self.model = model
        self.conf = 0.25
        self.iou = 0.5
        self.imgsz = 640
        self.device = "cpu"
        self.frame_stride = 1
        self.mirror = True
        self._i = 0
        self._lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if self.mirror:
            img = cv2.flip(img, 1)

        out = img
        # Throttle inference if desired
        do_infer = (self._i % max(int(self.frame_stride), 1) == 0)
        self._i += 1

        if do_infer:
            with self._lock:
                results = self.model.predict(
                    source=img,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    device=self.device if self.device else None,
                    verbose=False
                )
            out = results[0].plot()  # BGR

        return av.VideoFrame.from_ndarray(out, format="bgr24")

with tab_cam:
    st.markdown("**Live webcam inference (browser)** — works locally and on Streamlit Cloud.")
    cam_cols = st.columns(3)
    mirror = cam_cols[0].toggle("Mirror", value=True)
    cam_stride = cam_cols[1].number_input("Frame stride", min_value=1, max_value=5, value=1, step=1, help="Process every Nth frame to improve FPS.")
    cam_imgsz = cam_cols[2].select_slider("Image size", options=[320, 416, 512, 640, 768, 960], value=imgsz)

    # WebRTC setup (use a public STUN server)
    rtc_cfg = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # Factory to inject current model into the processor
    def processor_factory():
        return YOLOWebcamProcessor(model)

    ctx = webrtc_streamer(
        key="yolov8-ppe-webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_cfg,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=processor_factory,
        async_processing=True,
    )

    # Live-update processor parameters from sidebar controls
    if ctx.video_processor:
        vp: YOLOWebcamProcessor = ctx.video_processor
        vp.conf = float(conf)
        vp.iou = float(iou)
        vp.imgsz = int(cam_imgsz)
        vp.device = device_choice
        vp.frame_stride = int(cam_stride)
        vp.mirror = bool(mirror)

    st.caption(
        "Notes: WebRTC runs in-browser; GPU is used only if the server hosting this app has a CUDA device. "
        "On Streamlit Cloud the session is CPU-only—consider increasing frame stride."
    )

"""
streamlit_app.py
----------------
Real-time ANPR (Automatic Number Plate Recognition).

Two input modes, in tabs:
  1. Upload Video  — process a video file frame by frame.
  2. Live Camera   — continuous live webcam feed, processed in real time
                      (local machine only — needs a physical camera, so
                      this tab will not work on a cloud deployment without
                      one).

Both modes: detect vehicles, locate plates, read text with OCR, show a
live annotated preview, keep a running log of every plate seen, and raise
an alert whenever a plate matches a user-defined watchlist.

Run locally:
    streamlit run streamlit_app.py

Deploy:
    Push this folder to GitHub (keep the utils/ folder structure intact!)
    then deploy on share.streamlit.io pointing at streamlit_app.py.
    Note: the Live Camera tab needs a physical camera attached to wherever
    the app runs, so it's only meaningful when you run this locally.
"""

from __future__ import annotations

import queue
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

from pipeline import run_pipeline, annotate_image


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ANPR — Real-time Plate Recognition",
    page_icon="🚓",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_plate(text: str) -> str:
    """Normalize a plate string for matching: uppercase, strip anything
    that isn't a letter or digit (spaces, dashes, etc.)."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


@st.cache_resource(show_spinner=False)
def warm_up(model_name: str, langs: tuple[str, ...]):
    """Load the YOLO model + EasyOCR reader once and cache for the session."""
    from utils.detector import _get_model
    from utils.ocr import _get_reader

    _get_model(model_name)
    _get_reader(list(langs))
    return True


@dataclass
class Sighting:
    frame: int
    timestamp_s: float
    plate_text: str
    plate_norm: str
    confidence: float
    is_watchlisted: bool


@dataclass
class SessionStats:
    sightings: list[Sighting] = field(default_factory=list)
    seen_norms: set[str] = field(default_factory=set)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Live camera support (streamlit-webrtc)
# ---------------------------------------------------------------------------
# webrtc_streamer runs video frame callbacks on a background thread, separate
# from the main Streamlit script thread. We can't safely mutate
# st.session_state directly from that thread, so the processor pushes new
# Sighting objects onto a thread-safe queue, and the main thread drains that
# queue on each rerun to update the UI.

class LivePlateProcessor(VideoProcessorBase):
    """streamlit-webrtc video frame callback: runs the ANPR pipeline on a
    throttled subset of incoming frames and pushes results to a queue that
    the main thread reads from."""

    def __init__(self) -> None:
        self.model_name = "yolov8n.pt"
        self.conf_threshold = 0.35
        self.ocr_languages: list[str] = ["en"]
        self.max_vehicles = 3
        self.sample_every_n = 5
        self.watchlist: set[str] = set()

        self._frame_idx = 0
        self.result_queue: "queue.Queue" = queue.Queue(maxsize=50)
        self._lock = threading.Lock()
        self._last_annotated: np.ndarray | None = None

    def update_settings(
        self, model_name, conf_threshold, ocr_languages, max_vehicles,
        sample_every_n, watchlist,
    ):
        with self._lock:
            self.model_name = model_name
            self.conf_threshold = conf_threshold
            self.ocr_languages = ocr_languages
            self.max_vehicles = max_vehicles
            self.sample_every_n = max(1, sample_every_n)
            self.watchlist = watchlist

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self._frame_idx += 1

        with self._lock:
            sample_every_n = self.sample_every_n
            model_name = self.model_name
            conf_threshold = self.conf_threshold
            ocr_languages = list(self.ocr_languages)
            max_vehicles = self.max_vehicles
            watchlist = set(self.watchlist)

        if self._frame_idx % sample_every_n == 0:
            try:
                results = run_pipeline(
                    img,
                    model_name=model_name,
                    conf_threshold=conf_threshold,
                    ocr_languages=ocr_languages,
                    max_vehicles=max_vehicles,
                )
                annotated = annotate_image(img, results)
                self._last_annotated = annotated

                for r in results:
                    if not r.plate_text:
                        continue
                    norm = normalize_plate(r.plate_text)
                    if not norm:
                        continue
                    sighting = Sighting(
                        frame=self._frame_idx,
                        timestamp_s=time.time(),
                        plate_text=r.plate_text,
                        plate_norm=norm,
                        confidence=r.vehicle.confidence,
                        is_watchlisted=norm in watchlist,
                    )
                    try:
                        self.result_queue.put_nowait(sighting)
                    except queue.Full:
                        pass
            except Exception:
                # Never let a pipeline error kill the video stream.
                annotated = img
        else:
            annotated = self._last_annotated if self._last_annotated is not None else img

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------

st.sidebar.title("⚙️ Settings")

model_name = st.sidebar.selectbox(
    "YOLOv8 model",
    options=["yolov8n.pt", "yolov8s.pt"],
    index=0,
    help="Nano (n) is fastest and is bundled with this app. Small (s) is "
         "more accurate but downloads on first use.",
)

conf_threshold = st.sidebar.slider(
    "Detection confidence threshold",
    min_value=0.05, max_value=0.95, value=0.35, step=0.05,
)

lang_options = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Portuguese": "pt", "Arabic": "ar", "Hindi": "hi",
    "Chinese (Simplified)": "ch_sim",
}
lang_labels = st.sidebar.multiselect(
    "OCR language(s)", options=list(lang_options.keys()), default=["English"],
)
ocr_languages = [lang_options[label] for label in lang_labels] or ["en"]

max_vehicles = st.sidebar.slider("Max vehicles per frame", 1, 10, 3)

st.sidebar.markdown("---")
st.sidebar.subheader("🎞️ Processing speed")
sample_every_n = st.sidebar.slider(
    "Process every Nth frame", min_value=1, max_value=30, value=10,
    help="Running detection+OCR on every single frame is slow on CPU. "
         "Skipping frames trades temporal resolution for speed — a value "
         "of 10 means roughly 3 processed frames per second of a 30fps "
         "video.",
)
max_frames = st.sidebar.number_input(
    "Max frames to process (0 = no limit)", min_value=0, value=300, step=50,
    help="Caps total processing time. 300 processed frames at sample-every-10 "
         "covers ~100 seconds of 30fps footage.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("🚨 Watchlist")
watchlist_raw = st.sidebar.text_area(
    "Plates to flag (one per line)",
    placeholder="ABC123\nXYZ-789",
    help="Matching is exact (after normalizing case/spaces/dashes), not fuzzy. "
         "OCR misreads (e.g. 0 vs O) will not match.",
)
watchlist = {normalize_plate(line) for line in watchlist_raw.splitlines() if line.strip()}

st.sidebar.markdown("---")
st.sidebar.caption(
    "First run loads the YOLO and EasyOCR models — this can take 30–60s. "
    "Later runs in the same session are fast."
)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🚓 Real-time License Plate Recognition")
st.write(
    "Process footage of moving vehicles — upload a video file, or use your "
    "live camera. Either way: vehicle detection → plate localization → "
    "OCR, with watchlist alerts for flagged plates."
)

# ---------------------------------------------------------------------------
# Session state for results (persists across reruns within a session)
# ---------------------------------------------------------------------------

if "stats" not in st.session_state:
    st.session_state.stats = SessionStats()

reset = st.button("🔄 Clear results")
if reset:
    st.session_state.stats = SessionStats()
    st.rerun()


def render_table(placeholder):
    stats: SessionStats = st.session_state.stats
    if not stats.sightings:
        placeholder.info("No plates logged yet.")
        return
    df = pd.DataFrame(
        [
            {
                "Frame": s.frame,
                "Time (s)": round(s.timestamp_s, 1),
                "Plate": s.plate_text,
                "Watchlist match": "🚨 YES" if s.is_watchlisted else "",
            }
            for s in reversed(stats.sightings)
        ]
    )
    placeholder.dataframe(df, use_container_width=True, hide_index=True)


def download_log_button(key_suffix: str):
    if st.session_state.stats.sightings:
        df_full = pd.DataFrame(
            [
                {
                    "frame": s.frame,
                    "timestamp_s": round(s.timestamp_s, 2),
                    "plate_text": s.plate_text,
                    "plate_normalized": s.plate_norm,
                    "watchlist_match": s.is_watchlisted,
                }
                for s in st.session_state.stats.sightings
            ]
        )
        st.download_button(
            "⬇️ Download full log as CSV",
            data=df_full.to_csv(index=False).encode("utf-8"),
            file_name="plate_log.csv",
            mime="text/csv",
            key=f"download_{key_suffix}",
        )


# ---------------------------------------------------------------------------
# Tab 1 — Upload Video
# ---------------------------------------------------------------------------

with tab_video:
    uploaded_video = st.file_uploader(
        "Upload a video file", type=["mp4", "mov", "avi", "mkv", "webm"]
    )
    start = st.button("▶️ Start processing", type="primary", disabled=uploaded_video is None)

    alert_placeholder = st.empty()
    video_placeholder = st.empty()
    progress_placeholder = st.empty()
    table_placeholder = st.empty()
    render_table(table_placeholder)

    if start and uploaded_video is not None:
        with st.spinner("Loading models (first run only)…"):
            warm_up(model_name, tuple(ocr_languages))

        # Write upload to a temp file so OpenCV can open it by path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Could not open this video file. Try a different format (mp4 recommended).")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

            frame_idx = 0
            processed_count = 0
            t_start = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                if frame_idx % sample_every_n != 0:
                    continue

                results = run_pipeline(
                    frame,
                    model_name=model_name,
                    conf_threshold=conf_threshold,
                    ocr_languages=ocr_languages,
                    max_vehicles=max_vehicles,
                )
                annotated = annotate_image(frame, results)
                timestamp_s = frame_idx / fps

                new_alerts = []
                for r in results:
                    if not r.plate_text:
                        continue
                    norm = normalize_plate(r.plate_text)
                    if not norm:
                        continue
                    is_match = norm in watchlist
                    sighting = Sighting(
                        frame=frame_idx,
                        timestamp_s=timestamp_s,
                        plate_text=r.plate_text,
                        plate_norm=norm,
                        confidence=r.vehicle.confidence,
                        is_watchlisted=is_match,
                    )
                    st.session_state.stats.sightings.append(sighting)
                    st.session_state.stats.seen_norms.add(norm)
                    if is_match:
                        new_alerts.append(r.plate_text)

                if new_alerts:
                    alert_placeholder.error(
                        f"🚨 WATCHLIST MATCH at {timestamp_s:.1f}s (frame {frame_idx}): "
                        + ", ".join(new_alerts)
                    )

                video_placeholder.image(
                    bgr_to_rgb(annotated),
                    caption=f"Frame {frame_idx}" + (f" / {total_frames}" if total_frames else "")
                            + f"  ·  t={timestamp_s:.1f}s",
                    use_container_width=True,
                )

                processed_count += 1
                if total_frames:
                    progress_placeholder.progress(
                        min(frame_idx / total_frames, 1.0),
                        text=f"Processed {processed_count} frame(s) — "
                             f"{frame_idx}/{total_frames} read",
                    )
                else:
                    progress_placeholder.text(f"Processed {processed_count} frame(s)…")

                render_table(table_placeholder)

                if max_frames and processed_count >= max_frames:
                    st.warning(f"Stopped after reaching the {max_frames}-frame processing cap.")
                    break

            cap.release()
            elapsed = time.time() - t_start
            st.success(
                f"Done. Processed {processed_count} frame(s) in {elapsed:.1f}s "
                f"({len(st.session_state.stats.sightings)} plate reading(s), "
                f"{len(st.session_state.stats.seen_norms)} unique plate(s))."
            )
            download_log_button("video")

    elif uploaded_video is None:
        st.info("Upload a video above, set your watchlist in the sidebar, then click **Start processing**.")


# ---------------------------------------------------------------------------
# Tab 2 — Live Camera
# ---------------------------------------------------------------------------

with tab_live:
    st.caption(
        "Uses a physical camera attached to the machine running this app "
        "(via your browser). This only works when you run the app "
        "**locally** — a cloud deployment has no camera to access."
    )

    with st.spinner("Loading models (first run only)…"):
        warm_up(model_name, tuple(ocr_languages))

    webrtc_ctx = webrtc_streamer(
        key="live-anpr",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=LivePlateProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.update_settings(
            model_name=model_name,
            conf_threshold=conf_threshold,
            ocr_languages=ocr_languages,
            max_vehicles=max_vehicles,
            sample_every_n=sample_every_n,
            watchlist=watchlist,
        )

    live_alert_placeholder = st.empty()
    live_table_placeholder = st.empty()
    render_table(live_table_placeholder)

    if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
        # Drain any new sightings pushed by the background video thread and
        # fold them into session state, which lives on the main thread.
        drained_any = False
        new_alerts = []
        while True:
            try:
                sighting = webrtc_ctx.video_processor.result_queue.get_nowait()
            except queue.Empty:
                break
            st.session_state.stats.sightings.append(sighting)
            st.session_state.stats.seen_norms.add(sighting.plate_norm)
            if sighting.is_watchlisted:
                new_alerts.append(sighting.plate_text)
            drained_any = True

        if new_alerts:
            live_alert_placeholder.error(
                "🚨 WATCHLIST MATCH: " + ", ".join(new_alerts)
            )

        if drained_any:
            render_table(live_table_placeholder)

        # Keep polling the queue for new sightings while the stream is live.
        time.sleep(0.5)
        st.rerun()

    download_log_button("live")


"""
streamlit_app.py
----------------
Real-time(ish) ANPR (Automatic Number Plate Recognition) over video.

Upload a video file (dashcam / traffic-cam footage). The app processes it
frame by frame: detects vehicles, locates plates, reads the text with OCR,
shows a live annotated preview as it goes, keeps a running log of every
plate seen, and raises an alert whenever a plate matches a user-defined
watchlist.

Run locally:
    streamlit run streamlit_app.py

Deploy:
    Push this folder to GitHub (keep the utils/ folder structure intact!)
    then deploy on share.streamlit.io pointing at streamlit_app.py.
"""

from __future__ import annotations

import re
import tempfile
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd
import streamlit as st

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
    "Upload footage of moving vehicles (dashcam, traffic camera, etc.). "
    "The app samples frames, runs vehicle detection → plate localization → "
    "OCR on each, and flags any plate that matches your watchlist."
)

uploaded_video = st.file_uploader(
    "Upload a video file", type=["mp4", "mov", "avi", "mkv", "webm"]
)


# ---------------------------------------------------------------------------
# Session state for results (persists across reruns within a session)
# ---------------------------------------------------------------------------

if "stats" not in st.session_state:
    st.session_state.stats = SessionStats()

start = st.button("▶️ Start processing", type="primary", disabled=uploaded_video is None)
reset = st.button("🔄 Clear results")

if reset:
    st.session_state.stats = SessionStats()
    st.rerun()


# ---------------------------------------------------------------------------
# Live processing
# ---------------------------------------------------------------------------

alert_placeholder = st.empty()
video_placeholder = st.empty()
progress_placeholder = st.empty()
table_placeholder = st.empty()

def render_table():
    stats: SessionStats = st.session_state.stats
    if not stats.sightings:
        table_placeholder.info("No plates logged yet.")
        return
    df = pd.DataFrame(
        [
            {
                "Frame": s.frame,
                "Time (s)": round(s.timestamp_s, 1),
                "Plate": s.plate_text,
                "Confidence-filtered": "✅" if s.confidence > 0 else "—",
                "Watchlist match": "🚨 YES" if s.is_watchlisted else "",
            }
            for s in reversed(stats.sightings)
        ]
    )
    table_placeholder.dataframe(df, use_container_width=True, hide_index=True)

render_table()

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

            render_table()

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
            )

elif uploaded_video is None:
    st.info("Upload a video above, set your watchlist in the sidebar, then click **Start processing**.")

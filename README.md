# Real-time License Plate Recognition (ANPR) — Streamlit App

Upload a video of moving vehicles (dashcam, traffic camera footage, etc.).
The app processes it frame by frame: detects vehicles with YOLOv8, locates
the license plate, reads it with EasyOCR, shows a live annotated preview as
it goes, keeps a running log of every plate seen, and raises an alert when a
plate matches a watchlist you define.

This is a Streamlit front-end for the original CLI pipeline
(`pipeline.py` + `utils/`) from the
[Real-time-License-Plate-Recognition](https://github.com/Omer684/Real-time-License-Plate-Recognition)
project. The detection/OCR logic is unchanged — only the interface is new.

## ⚠️ Folder structure matters

This project **requires** the helper modules to live inside a folder
literally named `utils/`, because `pipeline.py` imports them as
`from utils.detector import ...`, etc. When pushing to GitHub, make sure
your repo looks like this and not flattened into one folder:

```
your-repo/
├── streamlit_app.py
├── pipeline.py
├── requirements.txt
├── packages.txt
├── yolov8n.pt
└── utils/                 <-- must be an actual folder
    ├── __init__.py
    ├── detector.py
    ├── plate_locator.py
    ├── ocr.py
    └── image_utils.py
```

If you upload files individually through GitHub's web UI, use "Add file →
Upload files" and drag the whole `utils` folder in one go (or create the
folder first by naming a file `utils/detector.py` when uploading) — don't
drop the contents of `utils/` directly into the repo root.

## Requirements

- Python 3.10+ (the codebase uses `X | None` type-hint syntax)

## Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run streamlit_app.py
```

The first run loads the YOLOv8 and EasyOCR models, which can take 30–60
seconds. The `yolov8n.pt` weights are already bundled so that step is
skipped for the default model.

## How to use it

1. **Upload a video** — dashcam clips or traffic-camera footage work well.
   You can find free sample footage on sites like Pexels or Pixabay if you
   don't have your own ("traffic", "highway", "dashcam" are good search
   terms) — just check the license terms before using any clip publicly.
2. **(Optional) Set a watchlist** in the sidebar — one plate per line.
   Matching is **exact**, after normalizing case and stripping spaces/dashes
   (so `abc-123` and `ABC 123` are treated the same). OCR misreads (e.g.
   `0` read as `O`) will *not* match — this is a deliberate choice to avoid
   false alerts.
3. **Tune processing speed** — running detection+OCR on every frame is slow
   on CPU, so the app only processes every Nth frame (configurable in the
   sidebar). A frame cap also limits total runtime, useful when testing on
   Streamlit Cloud's shared CPU resources.
4. **Click "Start processing"** — watch the live annotated frame, the
   running plate log, and any watchlist alerts as they appear.
5. **Download the full CSV log** once processing finishes.

## Deploy to Streamlit Community Cloud (free)

1. Push this folder's contents to a **public** GitHub repo, preserving the
   `utils/` folder (see warning above).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **"New app"**, pick your repo/branch, set the main file path to
   `streamlit_app.py`, and deploy.
4. First boot is slow (installing `torch`, `ultralytics`, `easyocr`).
   Streamlit Cloud's free tier has limited CPU/RAM — keep "Max frames to
   process" and "Process every Nth frame" conservative there, especially
   for longer videos.

### Why some files differ from a typical CLI setup

- **`opencv-python` → `opencv-python-headless`**: Streamlit Cloud runs
  headless Linux containers with no display server; the regular
  `opencv-python` package depends on GUI libraries that often fail to
  install there.
- **`packages.txt`** installs `libgl1`, a system library `opencv`/`torch`
  need even in headless mode.
- **No live webcam / RTSP support.** A continuous camera loop needs direct
  access to a physical or networked camera device, which a shared cloud
  server doesn't have, and doesn't fit Streamlit's rerun-on-interaction
  execution model even when run locally. Uploading a video file instead
  gives the same "process moving footage frame-by-frame" behavior without
  needing dedicated hardware — if you later deploy this on real hardware
  (e.g. a Raspberry Pi or NUC next to a road) with an RTSP camera, the
  same `pipeline.py` functions can be wired into a `cv2.VideoCapture(rtsp_url)`
  loop instead of a file path.
- **No persistent database.** Streamlit Cloud's filesystem is ephemeral, so
  the plate log lives only in the browser session (`st.session_state`) and
  is offered as a CSV download instead of being written to a database.

## App structure

```
.
├── streamlit_app.py     # Streamlit UI — video upload, live processing, watchlist (new)
├── pipeline.py          # Detection + OCR orchestration (unchanged)
├── utils/
│   ├── detector.py       # YOLOv8 vehicle detection (unchanged)
│   ├── plate_locator.py  # Plate localization within a vehicle crop (unchanged)
│   ├── ocr.py             # EasyOCR wrapper (unchanged)
│   ├── image_utils.py     # Crop/resize/annotate helpers (unchanged)
│   └── database.py        # Unused by the Streamlit app; kept for reference
├── requirements.txt
├── packages.txt           # System packages for Streamlit Cloud
└── yolov8n.pt              # Pre-downloaded YOLOv8 nano weights
```

## Possible CV/resume framing

> Built a real-time Automatic Number Plate Recognition (ANPR) pipeline
> (YOLOv8 vehicle detection → plate localization → EasyOCR text
> recognition) with a Streamlit interface for frame-by-frame video
> processing, live annotated preview, and exact-match watchlist alerting
> for flagged plates. Deployed to Streamlit Community Cloud.

# AI Phone Detector Pro - macOS Apple Silicon

Real-time phone detection for live streams and cameras, built specifically for Apple Silicon Macs. This edition targets Apple Metal through PyTorch MPS and intentionally keeps Windows/CUDA code out of the macOS path.

## Features

- Apple Silicon only: M1, M2, M3, M4 M5 and newer
- Metal/MPS acceleration for YOLOv8 inference
- RTSP, HTTP/HLS, YouTube, Twitch and TikTok stream input
- Twitch extraction through Streamlink with yt-dlp fallback
- PyQt6 desktop interface with live preview and detection stats
- Person and cell-phone detection with alert levels
- Optional Core ML export helper for Neural Engine experiments

## Apple Neural Engine

The live app uses PyTorch MPS, which runs on Apple GPU cores through Metal. That is the practical real-time path for YOLO in Python.

The Apple Neural Engine is not directly exposed to PyTorch. To use ANE, export the YOLO model to Core ML and run it through a dedicated Core ML inference pipeline. This release includes `export_coreml_model.py` as a starting point, but live detection currently uses MPS for reliability.

## Requirements

- Apple Silicon Mac, arm64 only
- macOS with Metal/MPS support
- Python 3.11 or 3.12 arm64
- Internet connection on first model download

## Install

```bash
git clone github.com/NNW-Studios/AI-phone-Detector
cd <your-repo-folder>
chmod +x INSTALL_MAC.sh
./INSTALL_MAC.sh
```

The installer creates a local `venv/`, installs dependencies, checks MPS, and generates `START_MAC.sh`.

## Run

```bash
./START_MAC.sh
```

On first run, Ultralytics may download the selected YOLO model, for example `yolov8s.pt`. Model files are ignored by Git and should not be committed.

## Twitch Notes

Use a channel URL like:

```text
https://www.twitch.tv/channelname
```

If Twitch extraction fails, update the stream extractors:

```bash
source venv/bin/activate
python -m pip install -U streamlink yt-dlp
```

## Recommended Models

- MacBook Air M1/M2: Nano or Small
- MacBook Pro M1 Pro/M2 Pro/M3 Pro: Small or Medium
- M1 Max/M2 Max/M3 Max/Ultra/M4/M5: Medium or Large

## Core ML Export

```bash
source venv/bin/activate
python export_coreml_model.py --model yolov8n.pt
```

This creates a Core ML package next to the YOLO model. Integrating that package into live detection requires a separate Core ML runtime path.


 License

MIT License. See `LICENSE`.

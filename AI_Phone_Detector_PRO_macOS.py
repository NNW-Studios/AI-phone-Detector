"""
AI PHONE DETECTOR PRO - macOS APPLE SILICON EDITION
===================================================
Features:
- Apple Silicon first: macOS ARM64 only
- Metal Performance Shaders (MPS) acceleration for M1/M2/M3/M4 GPUs
- Modern macOS dark UI
- TikTok Live stream support
- YouTube/Twitch stream analysis
- Advanced phone detection with YOLOv8
- Real-time performance monitoring

Note:
PyTorch uses Apple GPU cores through Metal/MPS. The Apple Neural Engine is only
available through Core ML, so this app keeps live YOLO inference on MPS and can
be paired with a Core ML export workflow later.

Author: AI Camera Solutions
Version: 4.1.0 APPLE SILICON
"""

import sys
import os
import platform
import cv2
import numpy as np

# Lets PyTorch safely route unsupported MPS operators to CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QTextEdit, QGroupBox, QGridLayout, QCheckBox,
                             QMessageBox, QComboBox, QSlider, QSpinBox, QFrame)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen, QLinearGradient
import time
from datetime import datetime
import re

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import yt_dlp
    YOUTUBE_SUPPORT = True
except ImportError:
    YOUTUBE_SUPPORT = False

try:
    import streamlink
    STREAMLINK_SUPPORT = True
except ImportError:
    STREAMLINK_SUPPORT = False


class StreamExtractionError(Exception):
    """Raised when stream extractors cannot provide a playable stream URL."""


def get_twitch_stream_url(url):
    """Extract Twitch live HLS URL with streamlink."""
    if not STREAMLINK_SUPPORT:
        raise StreamExtractionError("streamlink is not installed")

    try:
        streams = streamlink.streams(url)
    except Exception as e:
        raise StreamExtractionError(f"streamlink failed: {e}") from e

    if not streams:
        raise StreamExtractionError("streamlink found no playable Twitch streams; channel may be offline")

    for quality in ("best", "720p60", "720p", "480p", "worst"):
        stream = streams.get(quality)
        if stream:
            return stream.to_url()

    first_stream = next(iter(streams.values()))
    return first_stream.to_url()


def get_ytdlp_stream_url(url):
    """Extract direct stream URL with yt-dlp."""
    if not YOUTUBE_SUPPORT:
        raise StreamExtractionError("yt-dlp is not installed")

    try:
        ydl_opts = {
            'format': 'best',
            'quiet': True,
            'no_warnings': False,
            'noplaylist': True,
            'http_headers': {
                'User-Agent': (
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/124.0.0.0 Safari/537.36'
                )
            },
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        if not info:
            raise StreamExtractionError("yt-dlp returned no stream information")

        live_status = info.get('live_status')
        if live_status in ('is_upcoming', 'was_live', 'not_live'):
            raise StreamExtractionError(f"stream is not currently live ({live_status})")

        direct_url = info.get('url')
        if direct_url:
            return direct_url

        formats = info.get('formats') or []
        candidates = []
        for fmt in formats:
            stream_url = fmt.get('url')
            if not stream_url:
                continue
            if fmt.get('vcodec') == 'none':
                continue

            protocol = fmt.get('protocol') or ''
            ext = fmt.get('ext') or ''
            is_hls = 'm3u8' in protocol or stream_url.endswith('.m3u8')
            is_video_file = ext in ('mp4', 'webm', 'mov') or fmt.get('vcodec')
            if is_hls or is_video_file:
                candidates.append(fmt)

        if candidates:
            candidates.sort(key=lambda fmt: (fmt.get('height') or 0, fmt.get('tbr') or 0))
            return candidates[-1].get('url')

        available = ', '.join(
            sorted({str(fmt.get('format_id')) for fmt in formats if fmt.get('format_id')})
        )
        if available:
            raise StreamExtractionError(f"no playable video URL found; available formats: {available}")
        raise StreamExtractionError("no playable video formats found")
    except Exception as e:
        if isinstance(e, StreamExtractionError):
            raise
        raise StreamExtractionError(str(e)) from e


def get_stream_url(url, platform):
    """Extract direct stream URL from YouTube/Twitch/TikTok."""
    errors = []

    if platform == "Twitch":
        try:
            return get_twitch_stream_url(url)
        except StreamExtractionError as e:
            errors.append(f"streamlink: {e}")

    try:
        return get_ytdlp_stream_url(url)
    except StreamExtractionError as e:
        errors.append(f"yt-dlp: {e}")

    raise StreamExtractionError(" | ".join(errors))


class DetectionThread(QThread):
    """Enhanced AI Detection Thread optimized for Apple Silicon / Metal."""
    frame_ready = pyqtSignal(np.ndarray)
    stats_update = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    gpu_info_signal = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.stream_url = ""
        self.running = False
        self.model = None
        self.device = 'cpu'
        self.conf_threshold = 0.20
        self.detect_persons = True
        self.detect_phones = True
        self.use_fp16 = False
        self.model_size = 'n'
        self.gpu_info = {}
        
    def set_stream_url(self, url):
        self.stream_url = url
        
    def initialize_apple_silicon(self):
        """Initialize Apple Silicon Metal acceleration through PyTorch MPS."""
        try:
            self.status_signal.emit("🔍 Checking Apple Silicon / Metal availability...")

            machine = platform.machine().lower()
            system = platform.system()

            if system != "Darwin" or machine not in ("arm64", "aarch64"):
                self.status_signal.emit("❌ This macOS build is Apple Silicon only")
                self.device = 'cpu'
                self.gpu_info = {
                    'available': False,
                    'device': 'CPU',
                    'name': f'{system} {machine}',
                    'vram': 0,
                    'backend': 'Unsupported platform',
                    'pytorch_version': torch.__version__
                }
                self.gpu_info_signal.emit(self.gpu_info)
                return False

            mps_backend = getattr(torch.backends, "mps", None)
            mps_built = bool(mps_backend and torch.backends.mps.is_built())
            mps_available = bool(mps_backend and torch.backends.mps.is_available())

            if not mps_built or not mps_available:
                self.status_signal.emit("⚠️ Metal/MPS not available - CPU fallback active")
                self.device = 'cpu'
                self.gpu_info = {
                    'available': False,
                    'device': 'CPU',
                    'name': 'Apple Silicon CPU fallback',
                    'vram': 0,
                    'backend': 'MPS unavailable',
                    'pytorch_version': torch.__version__
                }
                self.gpu_info_signal.emit(self.gpu_info)
                return False

            self.device = 'mps'
            chip_name = platform.processor() or "Apple Silicon"

            self.status_signal.emit("✅ Metal Performance Shaders available")
            self.status_signal.emit(f"🎮 Accelerator: {chip_name} GPU via MPS")
            self.status_signal.emit("ℹ️ Neural Engine requires Core ML export; live mode uses Metal GPU")

            # Keep FP32 on MPS for best YOLO operator compatibility.
            self.use_fp16 = False

            self.status_signal.emit("🔧 Testing Metal GPU computation...")
            test_tensor = torch.randn(1000, 1000, device=self.device)
            result = torch.matmul(test_tensor, test_tensor)
            del test_tensor, result
            if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            self.status_signal.emit("✅ Metal computation test passed")

            pytorch_version = torch.__version__
            self.status_signal.emit(f"📦 PyTorch: {pytorch_version}")
            self.status_signal.emit("🔧 Backend: Apple Metal / MPS")

            self.gpu_info = {
                'available': True,
                'device': self.device,
                'name': chip_name,
                'vram': 0,
                'backend': 'Apple Metal / MPS',
                'pytorch_version': pytorch_version,
                'fp16': self.use_fp16,
                'neural_engine': 'Core ML export required'
            }
            self.gpu_info_signal.emit(self.gpu_info)

            self.status_signal.emit("🚀 Apple Silicon acceleration ready!")
            return True

        except Exception as e:
            self.status_signal.emit(f"❌ Apple Silicon initialization failed: {str(e)}")
            self.device = 'cpu'
            self.gpu_info = {
                'available': False,
                'device': 'CPU',
                'name': 'CPU (Metal failed)',
                'vram': 0,
                'backend': 'CPU fallback',
                'error': str(e)
            }
            self.gpu_info_signal.emit(self.gpu_info)
            return False
    
    def load_model(self):
        """Load YOLOv8 model with Apple Metal optimization."""
        try:
            if not YOLO_AVAILABLE:
                self.error_signal.emit("Ultralytics not installed!\n\nRun: pip install ultralytics")
                return False
            
            self.status_signal.emit("📦 Loading AI model...")
            
            accelerator_available = self.initialize_apple_silicon()
            
            # Load model
            model_name = f'yolov8{self.model_size}.pt'
            self.status_signal.emit(f"📥 Loading {model_name}...")
            
            self.model = YOLO(model_name)
            
            if accelerator_available and self.device != 'cpu':
                self.status_signal.emit(f"🎯 Moving model to {self.device}...")
                self.model.to(self.device)

                self.status_signal.emit("🔥 Warming up Apple GPU...")
                dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)

                for i in range(3):
                    _ = self.model.predict(
                        dummy_frame,
                        device=self.device,
                        half=self.use_fp16,
                        verbose=False,
                        conf=self.conf_threshold
                    )
                    self.status_signal.emit(f"Warmup {i+1}/3 complete")

                if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                    torch.mps.synchronize()
                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
                self.status_signal.emit("✅ Apple GPU warmup complete")
            else:
                self.status_signal.emit("ℹ️ Running on CPU fallback (slower performance)")

            self.status_signal.emit("✅ Model ready for detection!")
            return True
            
        except Exception as e:
            self.error_signal.emit(f"Model loading failed:\n{str(e)}")
            return False
    
    def connect_to_stream(self):
        """Connect to RTSP, YouTube, Twitch, or TikTok stream"""
        url = self.stream_url.strip()
        
        # Check if it's a streaming platform URL
        if any(domain in url.lower() for domain in ['youtube.com', 'youtu.be', 'twitch.tv', 'tiktok.com']):
            platform = "Unknown"
            if 'youtube.com' in url.lower() or 'youtu.be' in url.lower():
                platform = "YouTube"
            elif 'twitch.tv' in url.lower():
                platform = "Twitch"
            elif 'tiktok.com' in url.lower():
                platform = "TikTok"

            if platform == "Twitch":
                if not STREAMLINK_SUPPORT and not YOUTUBE_SUPPORT:
                    self.error_signal.emit(
                        "Twitch stream support not installed!\n\n"
                        "Install with: pip install streamlink yt-dlp"
                    )
                    return None
            elif not YOUTUBE_SUPPORT:
                self.error_signal.emit(
                    "Stream support not installed!\n\n"
                    "Install with: pip install yt-dlp"
                )
                return None
            
            self.status_signal.emit(f"🎥 Extracting {platform} stream URL...")
            try:
                stream_url = get_stream_url(url, platform)
            except StreamExtractionError as e:
                update_hint = (
                    "Try updating stream extractors with:\n"
                    "source venv/bin/activate\n"
                    "python -m pip install -U streamlink yt-dlp"
                )
                self.error_signal.emit(
                    f"Failed to extract {platform} stream!\n\n"
                    f"Reason: {e}\n\n"
                    f"{update_hint}"
                )
                return None
            
            self.status_signal.emit(f"✅ {platform} stream URL extracted")
            url = stream_url
        
        # Connect to stream
        self.status_signal.emit(f"🔌 Connecting to stream...")
        cap = cv2.VideoCapture(url)
        
        if not cap.isOpened():
            self.error_signal.emit(f"Failed to connect to stream!\n\nURL: {url}")
            return None
        
        # Get stream info
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        self.status_signal.emit(f"✅ Connected: {width}x{height} @ {fps}fps")
        return cap
    
    def run(self):
        """Main detection loop"""
        try:
            # Load model
            if not self.load_model():
                return
            
            # Connect to stream
            cap = self.connect_to_stream()
            if cap is None:
                return
            
            self.running = True
            self.status_signal.emit("🚀 Detection started!")
            
            # Detection loop
            frame_count = 0
            start_time = time.time()
            
            while self.running:
                ret, frame = cap.read()
                
                if not ret:
                    self.status_signal.emit("⚠️ Stream ended or connection lost")
                    break
                
                # Run detection
                results = self.model.predict(
                    frame,
                    device=self.device,
                    half=self.use_fp16,
                    verbose=False,
                    conf=self.conf_threshold,
                    classes=[0, 67] if self.detect_persons and self.detect_phones else 
                           ([0] if self.detect_persons else [67] if self.detect_phones else None)
                )
                
                # Process results
                persons_count = 0
                phones_count = 0
                critical_alerts = 0
                high_alerts = 0
                
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        if cls == 0:  # Person
                            persons_count += 1
                            color = (0, 255, 0)
                            label = f"Person {conf:.2f}"
                        elif cls == 67:  # Cell phone
                            phones_count += 1
                            if conf > 0.6:
                                critical_alerts += 1
                                color = (0, 0, 255)
                                label = f"PHONE {conf:.2f} CRITICAL"
                            elif conf > 0.35:
                                high_alerts += 1
                                color = (0, 165, 255)
                                label = f"PHONE {conf:.2f} HIGH"
                            else:
                                color = (0, 255, 255)
                                label = f"Phone {conf:.2f}"
                        else:
                            continue
                        
                        # Draw detection
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Add FPS overlay
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Emit frame and stats
                self.frame_ready.emit(frame)
                self.stats_update.emit({
                    'fps': fps,
                    'persons': persons_count,
                    'phones': phones_count,
                    'critical': critical_alerts,
                    'high': high_alerts
                })
                
                # Reset counter every 100 frames
                if frame_count >= 100:
                    frame_count = 0
                    start_time = time.time()
            
            cap.release()
            self.status_signal.emit("✅ Detection stopped")
            
        except Exception as e:
            self.error_signal.emit(f"Detection error:\n{str(e)}")
    
    def stop(self):
        """Stop detection"""
        self.running = False


class MainWindow(QMainWindow):
    """Professional Main Window with modern dark theme"""
    
    def __init__(self):
        super().__init__()
        self.detection_thread = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize modern UI"""
        self.setWindowTitle("AI Phone Detector Pro - Apple Silicon Edition")
        self.setGeometry(100, 100, 1600, 900)
        
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            QWidget {
                background-color: #1a1a1a;
                color: #e0e0e0;
                font-family: 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif;
                font-size: 10pt;
            }
            QGroupBox {
                background-color: #252525;
                border: 2px solid #404040;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 15px;
                font-weight: bold;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 5px 10px;
                background-color: #2d2d2d;
                border-radius: 4px;
            }
            QLabel {
                color: #e0e0e0;
                background-color: transparent;
            }
            QLineEdit {
                background-color: #2d2d2d;
                border: 2px solid #404040;
                border-radius: 6px;
                padding: 8px;
                color: #ffffff;
                selection-background-color: #0078d4;
            }
            QLineEdit:focus {
                border: 2px solid #0078d4;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #1084d8;
            }
            QPushButton:pressed {
                background-color: #006cc1;
            }
            QPushButton:disabled {
                background-color: #404040;
                color: #808080;
            }
            QComboBox {
                background-color: #2d2d2d;
                border: 2px solid #404040;
                border-radius: 6px;
                padding: 6px;
                color: #ffffff;
            }
            QComboBox:hover {
                border: 2px solid #0078d4;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 8px solid #e0e0e0;
                margin-right: 5px;
            }
            QCheckBox {
                color: #e0e0e0;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #404040;
                border-radius: 4px;
                background-color: #2d2d2d;
            }
            QCheckBox::indicator:checked {
                background-color: #0078d4;
                border-color: #0078d4;
            }
            QTextEdit {
                background-color: #1e1e1e;
                border: 2px solid #404040;
                border-radius: 6px;
                color: #e0e0e0;
                font-family: 'SF Mono', 'Menlo', monospace;
                font-size: 9pt;
                padding: 8px;
            }
        """)
        
        # Left side - Video display
        left = QVBoxLayout()
        
        # Video frame
        video_group = QGroupBox("📹 Live Stream")
        video_layout = QVBoxLayout()
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(960, 540)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #000000;
                border: 3px solid #404040;
                border-radius: 8px;
            }
        """)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText("⏸ Awaiting Stream...")
        video_layout.addWidget(self.video_label)
        
        video_group.setLayout(video_layout)
        left.addWidget(video_group)
        
        # Stats panel
        stats_group = QGroupBox("📊 Detection Statistics")
        stats_layout = QGridLayout()
        stats_layout.setSpacing(10)
        
        # Create stat displays
        stat_style = """
            QLabel {
                background-color: #2d2d2d;
                border: 2px solid #404040;
                border-radius: 6px;
                padding: 10px;
                font-size: 11pt;
            }
        """
        
        # FPS
        stats_layout.addWidget(QLabel("FPS:"), 0, 0)
        self.fps_label = QLabel("--")
        self.fps_label.setStyleSheet(stat_style + "QLabel { color: #4CAF50; font-weight: bold; }")
        stats_layout.addWidget(self.fps_label, 0, 1)
        
        # Persons
        stats_layout.addWidget(QLabel("Persons:"), 0, 2)
        self.persons_label = QLabel("0")
        self.persons_label.setStyleSheet(stat_style + "QLabel { color: #2196F3; font-weight: bold; }")
        stats_layout.addWidget(self.persons_label, 0, 3)
        
        # Phones
        stats_layout.addWidget(QLabel("Phones:"), 1, 0)
        self.phones_label = QLabel("0")
        self.phones_label.setStyleSheet(stat_style + "QLabel { color: #FFC107; font-weight: bold; }")
        stats_layout.addWidget(self.phones_label, 1, 1)
        
        # Critical
        stats_layout.addWidget(QLabel("Critical:"), 1, 2)
        self.critical_label = QLabel("0")
        self.critical_label.setStyleSheet(stat_style + "QLabel { color: #F44336; font-weight: bold; }")
        stats_layout.addWidget(self.critical_label, 1, 3)
        
        # High alerts
        stats_layout.addWidget(QLabel("High:"), 2, 0)
        self.high_label = QLabel("0")
        self.high_label.setStyleSheet(stat_style + "QLabel { color: #FF9800; font-weight: bold; }")
        stats_layout.addWidget(self.high_label, 2, 1)
        
        stats_group.setLayout(stats_layout)
        left.addWidget(stats_group)
        
        main_layout.addLayout(left, 2)
        
        # Right side - Controls
        right = QVBoxLayout()
        
        # Stream input
        input_group = QGroupBox("🌐 Stream Configuration")
        input_layout = QVBoxLayout()
        input_layout.setSpacing(10)
        
        # Stream type selector
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Platform:"))
        self.stream_type = QComboBox()
        self.stream_type.addItems([
            "RTSP Camera",
            "YouTube Live",
            "Twitch Stream",
            "TikTok Live",
            "HTTP Stream"
        ])
        self.stream_type.currentIndexChanged.connect(self.update_url_placeholder)
        type_layout.addWidget(self.stream_type)
        input_layout.addLayout(type_layout)
        
        # URL input
        input_layout.addWidget(QLabel("Stream URL:"))
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("rtsp://username:password@ip:port/stream")
        input_layout.addWidget(self.url_input)
        
        input_group.setLayout(input_layout)
        right.addWidget(input_group)
        
        # Model settings
        model_group = QGroupBox("🤖 AI Model Settings")
        model_layout = QVBoxLayout()
        model_layout.setSpacing(10)
        
        # Model size
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Model:"))
        self.model_selector = QComboBox()
        self.model_selector.addItems([
            "Nano (Fastest)",
            "Small (Balanced)",
            "Medium (Accurate)",
            "Large (Best)"
        ])
        self.model_selector.setCurrentIndex(1)
        size_layout.addWidget(self.model_selector)
        model_layout.addLayout(size_layout)
        
        # Detection options
        self.detect_persons_cb = QCheckBox("Detect Persons")
        self.detect_persons_cb.setChecked(True)
        model_layout.addWidget(self.detect_persons_cb)
        
        self.detect_phones_cb = QCheckBox("Detect Phones")
        self.detect_phones_cb.setChecked(True)
        model_layout.addWidget(self.detect_phones_cb)
        
        model_group.setLayout(model_layout)
        right.addWidget(model_group)
        
        # Control buttons
        control_group = QGroupBox("🎮 Controls")
        control_layout = QVBoxLayout()
        control_layout.setSpacing(10)
        
        self.start_btn = QPushButton("▶ START DETECTION")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                padding: 12px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.start_btn.clicked.connect(self.start_detection)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹ STOP DETECTION")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                padding: 12px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        control_group.setLayout(control_layout)
        right.addWidget(control_group)
        
        # GPU Status
        gpu_group = QGroupBox("💻 System Status")
        gpu_layout = QVBoxLayout()
        
        self.status_label = QLabel("Checking system...")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #2d2d2d;
                border: 2px solid #404040;
                border-radius: 6px;
                padding: 15px;
                font-family: 'SF Mono', 'Menlo', monospace;
                font-size: 9pt;
            }
        """)
        self.status_label.setWordWrap(True)
        gpu_layout.addWidget(self.status_label)
        
        gpu_group.setLayout(gpu_layout)
        right.addWidget(gpu_group)
        
        # System log
        log_group = QGroupBox("📋 System Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(250)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        right.addWidget(log_group)
        
        right.addStretch()
        
        # Footer
        footer = QLabel("AI Phone Detector Pro | macOS Apple Silicon | Metal/MPS Edition")
        footer.setStyleSheet("""
            QLabel {
                color: #808080;
                font-size: 9pt;
                padding: 8px;
                background-color: #252525;
                border: 1px solid #404040;
                border-radius: 4px;
            }
        """)
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.addWidget(footer)
        
        main_layout.addLayout(right, 1)
        
        # Initialize
        self.log("System initialized successfully")
        self.log("Awaiting configuration...")
        self.check_system()
    
    def update_url_placeholder(self):
        """Update URL placeholder based on selected platform"""
        platform = self.stream_type.currentText()
        
        placeholders = {
            "RTSP Camera": "rtsp://username:password@192.168.1.100:554/stream",
            "YouTube Live": "https://www.youtube.com/watch?v=VIDEO_ID",
            "Twitch Stream": "https://www.twitch.tv/CHANNEL_NAME",
            "TikTok Live": "https://www.tiktok.com/@username/live",
            "HTTP Stream": "http://example.com/stream.m3u8"
        }
        
        self.url_input.setPlaceholderText(placeholders.get(platform, "Enter stream URL"))
    
    def check_system(self):
        """Check system capabilities"""
        try:
            system = platform.system()
            machine = platform.machine().lower()
            mps_backend = getattr(torch.backends, "mps", None)
            mps_built = bool(mps_backend and torch.backends.mps.is_built())
            mps_available = bool(mps_backend and torch.backends.mps.is_available())

            if system == "Darwin" and machine in ("arm64", "aarch64") and mps_available:
                status_text = (
                    f"✅ APPLE SILICON READY\n\n"
                    f"Device: Apple Silicon\n"
                    f"Backend: Metal / MPS\n"
                    f"Neural Engine: Core ML export required\n"
                    f"PyTorch: {torch.__version__}\n\n"
                    f"Status: Ready for detection"
                )
                self.status_label.setStyleSheet(self.status_label.styleSheet() + 
                                               "QLabel { color: #4CAF50; }")
                self.log("✅ Apple Silicon detected")
                self.log("✅ Metal/MPS acceleration enabled")
                self.log("ℹ️ Neural Engine path needs Core ML export, not PyTorch live inference")
            else:
                reason = "Not Apple Silicon macOS"
                if system == "Darwin" and machine in ("arm64", "aarch64") and not mps_built:
                    reason = "PyTorch was installed without MPS support"
                elif system == "Darwin" and machine in ("arm64", "aarch64") and not mps_available:
                    reason = "MPS is not available in this Python environment"

                status_text = (
                    "⚠️ APPLE METAL NOT READY\n\n"
                    f"Reason: {reason}\n"
                    "Mode: CPU fallback\n"
                    "Performance: Reduced"
                )
                self.status_label.setStyleSheet(self.status_label.styleSheet() + 
                                               "QLabel { color: #FFC107; }")
                self.log(f"⚠️ Metal/MPS unavailable: {reason}")
            
            self.status_label.setText(status_text)
            
            # Check dependencies
            if YOUTUBE_SUPPORT:
                self.log("✅ YouTube/Twitch/TikTok support enabled")
            else:
                self.log("⚠️ Stream support disabled (pip install yt-dlp)")

            if STREAMLINK_SUPPORT:
                self.log("✅ Twitch Streamlink support enabled")
            else:
                self.log("⚠️ Twitch Streamlink fallback disabled (pip install streamlink)")
            
            if YOLO_AVAILABLE:
                self.log("✅ YOLO AI model available")
            else:
                self.log("❌ YOLO not available (pip install ultralytics)")
            
        except Exception as e:
            self.log(f"❌ System check error: {e}")
    
    def log(self, msg):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {msg}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def start_detection(self):
        """Start detection process"""
        url = self.url_input.text().strip()
        
        if not url:
            QMessageBox.warning(self, "Input Required", "Please enter a stream URL")
            return
        
        if not YOLO_AVAILABLE:
            QMessageBox.critical(self, "Error", "Ultralytics not installed!\n\nRun: pip install ultralytics")
            return
        
        platform = self.stream_type.currentText()
        if any(x in platform for x in ["YouTube", "Twitch", "TikTok"]):
            if "Twitch" in platform and not STREAMLINK_SUPPORT and not YOUTUBE_SUPPORT:
                QMessageBox.warning(self, "Warning", 
                                   "Twitch support not installed!\n\nInstall: pip install streamlink yt-dlp")
                return
            if "Twitch" not in platform and not YOUTUBE_SUPPORT:
                QMessageBox.warning(self, "Warning", 
                                   f"{platform} support not installed!\n\nInstall: pip install yt-dlp")
                return
        
        self.log("=" * 60)
        self.log(f"🚀 Starting detection on {platform}")
        self.log(f"🎯 Target: {url}")
        
        # Create detection thread
        self.detection_thread = DetectionThread()
        self.detection_thread.set_stream_url(url)
        
        # Set model size
        model_idx = self.model_selector.currentIndex()
        model_sizes = ['n', 's', 'm', 'l']
        self.detection_thread.model_size = model_sizes[model_idx]
        
        # Connect signals
        self.detection_thread.frame_ready.connect(self.update_frame)
        self.detection_thread.stats_update.connect(self.update_stats)
        self.detection_thread.error_signal.connect(self.handle_error)
        self.detection_thread.status_signal.connect(self.log)
        self.detection_thread.gpu_info_signal.connect(self.update_gpu_info)
        
        # Apply settings
        self.detection_thread.detect_persons = self.detect_persons_cb.isChecked()
        self.detection_thread.detect_phones = self.detect_phones_cb.isChecked()
        
        # Start thread
        self.detection_thread.start()
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.url_input.setEnabled(False)
        self.stream_type.setEnabled(False)
        self.model_selector.setEnabled(False)
    
    def stop_detection(self):
        """Stop detection process"""
        if self.detection_thread:
            self.log("🛑 Stopping detection...")
            self.detection_thread.stop()
            self.detection_thread.wait(3000)
            if self.detection_thread.isRunning():
                self.detection_thread.terminate()
            self.detection_thread = None
        
        # Reset UI
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.url_input.setEnabled(True)
        self.stream_type.setEnabled(True)
        self.model_selector.setEnabled(True)
        
        self.video_label.clear()
        self.video_label.setText("⏸ Detection Stopped")
        
        # Reset stats
        self.fps_label.setText("--")
        self.persons_label.setText("0")
        self.phones_label.setText("0")
        self.critical_label.setText("0")
        self.high_label.setText("0")
        
        self.log("✅ Detection stopped - System ready")
    
    def update_frame(self, frame):
        """Update video display"""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img)
            scaled = pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.video_label.setPixmap(scaled)
        except Exception as e:
            self.log(f"❌ Frame update error: {e}")
    
    def update_stats(self, stats):
        """Update statistics display"""
        self.fps_label.setText(f"{stats['fps']:.1f}")
        self.persons_label.setText(str(stats['persons']))
        self.phones_label.setText(str(stats['phones']))
        self.critical_label.setText(str(stats['critical']))
        self.high_label.setText(str(stats['high']))
        
        # Log alerts
        if stats['critical'] > 0:
            self.log(f"🚨 CRITICAL: {stats['critical']} phone(s) detected!")
        elif stats['high'] > 0:
            self.log(f"⚠️ HIGH ALERT: {stats['high']} phone(s) detected")
    
    def update_gpu_info(self, info):
        """Update accelerator information display."""
        if info['available']:
            status_text = (
                f"✅ APPLE ACCELERATOR ACTIVE\n\n"
                f"Device: {info['name']}\n"
                f"Backend: {info['backend']}\n"
                f"Device target: {info['device']}\n"
                f"PyTorch: {info['pytorch_version']}\n"
                f"Neural Engine: {info.get('neural_engine', 'Core ML required')}\n"
            )
            status_text += "\nPrecision: FP32 for MPS compatibility"
            
            self.status_label.setStyleSheet(self.status_label.styleSheet() + 
                                           "QLabel { color: #4CAF50; }")
        else:
            status_text = (
                f"⚠️ CPU MODE\n\n"
                f"Device: {info['name']}\n"
                f"Backend: {info.get('backend', 'CPU')}\n"
                f"PyTorch: {info.get('pytorch_version', 'Unknown')}\n\n"
                f"Status: Limited performance"
            )
            self.status_label.setStyleSheet(self.status_label.styleSheet() + 
                                           "QLabel { color: #FFC107; }")
        
        self.status_label.setText(status_text)
    
    def handle_error(self, msg):
        """Handle errors"""
        self.log(f"❌ ERROR: {msg}")
        QMessageBox.critical(self, "Error", msg)
        self.stop_detection()
    
    def closeEvent(self, event):
        """Handle window close"""
        if self.detection_thread:
            self.detection_thread.stop()
            self.detection_thread.wait(2000)
            if self.detection_thread.isRunning():
                self.detection_thread.terminate()
        event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application metadata
    app.setApplicationName("AI Phone Detector Pro - Apple Silicon Edition")
    app.setApplicationVersion("4.0.0")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

@echo off
cls
echo ===============================================
echo  AI PHONE DETECTOR - Installation
echo ===============================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found
    echo Install from: python.org
    pause
    exit /b 1
)

python --version
echo.

if exist venv (
    echo Using existing venv...
) else (
    echo Creating venv...
    python -m venv venv
)

call venv\Scripts\activate.bat
echo.

echo Installing packages...
echo This takes 10-15 minutes
echo.

python -m pip install --upgrade pip --quiet

echo [1/6] PyTorch...
pip install torch torchvision torchaudio
echo Done

echo [2/6] OpenCV...
pip install opencv-python
echo Done

echo [3/6] PyQt6...
pip install PyQt6
echo Done

echo [4/6] NumPy...
pip install numpy
echo Done

echo [5/6] Ultralytics...
pip install ultralytics --no-deps
pip install matplotlib pillow pyyaml requests scipy tqdm psutil pandas seaborn
echo Done

echo [6/6] YouTube Support...
pip install yt-dlp
echo Done

echo.
echo ===============================================
echo Testing...
echo ===============================================
python -c "import torch; print('PyTorch OK')"
python -c "import cv2; print('OpenCV OK')"
python -c "import PyQt6; print('PyQt6 OK')"
python -c "from ultralytics import YOLO; print('YOLO OK')"

echo.
echo ===============================================
echo GPU Check
echo ===============================================
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo.
echo ===============================================
echo Installation Complete!
echo ===============================================
echo.

echo Creating START.bat...
echo @echo off > START.bat
echo cls >> START.bat
echo call venv\Scripts\activate.bat >> START.bat
echo python AI_Phone_Detector_PRO_v4.py >> START.bat
echo pause >> START.bat

echo Done!
echo.
echo Starting in 3 seconds...
timeout /t 3 >nul

cls
python AI_Phone_Detector_PRO_v4.py

pause

#!/usr/bin/env bash
set -euo pipefail

echo "==============================================="
echo " AI Phone Detector Pro - macOS Apple Silicon"
echo "==============================================="
echo

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: This installer is for macOS only."
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "ERROR: This build is Apple Silicon only."
  echo "Detected architecture: $(uname -m)"
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found."
  echo "Install Python 3.11 or 3.12 for macOS arm64 from python.org or Homebrew."
  exit 1
fi

PY_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$PY_VERSION" in
  3.11|3.12)
    ;;
  *)
    echo "ERROR: Python 3.11 or 3.12 is recommended for this release."
    echo "Detected Python: $PY_VERSION"
    echo "Install a native Apple Silicon Python 3.11/3.12, then rerun this installer."
    exit 1
    ;;
esac

PY_ARCH="$(python3 -c 'import platform; print(platform.machine())')"
if [[ "$PY_ARCH" != "arm64" ]]; then
  echo "ERROR: python3 is not running as arm64."
  echo "Detected Python architecture: $PY_ARCH"
  echo "Use a native Apple Silicon Python, not Rosetta."
  exit 1
fi

python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-macos.txt

python - <<'PY'
import platform
import torch

print("Python arch:", platform.machine())
print("PyTorch:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())

if not torch.backends.mps.is_available():
    raise SystemExit("ERROR: Metal/MPS is not available in this Python environment.")
PY

cat > START_MAC.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
python AI_Phone_Detector_PRO_macOS.py
SH
chmod +x START_MAC.sh

echo
echo "Installation complete."
echo "Start with: ./START_MAC.sh"

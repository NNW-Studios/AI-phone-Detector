# Contributing

This project is intentionally focused on Apple Silicon macOS. 

For windows Version go to github.com/NNW-Studios/AI-phone-Detector/releases/tag/ai

## Local Setup

```bash
chmod +x INSTALL_MAC.sh
./INSTALL_MAC.sh
./START_MAC.sh
```

## Pull Requests

- Keep Windows/CUDA changes out of this macOS edition.
- Do not commit downloaded model files or virtual environments.
- Prefer Metal/MPS-compatible PyTorch operations.
- Keep Core ML / Neural Engine work in a separate path from the live MPS detector.

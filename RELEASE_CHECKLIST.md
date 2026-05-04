# Release Checklist

- [ ] Commit only source files, scripts, docs, and metadata.
- [ ] Do not commit `venv/`.
- [ ] Do not commit `yolov8*.pt`; Ultralytics downloads models on first run.
- [ ] Run `python -m py_compile AI_Phone_Detector_PRO_macOS.py export_coreml_model.py`.
- [ ] Run `sh -n INSTALL_MAC.sh`.
- [ ] Test `./INSTALL_MAC.sh` on a fresh Apple Silicon Mac.
- [ ] Test one RTSP/HTTP stream and one Twitch live stream.
- [ ] Confirm the System Status panel says Metal/MPS is active.

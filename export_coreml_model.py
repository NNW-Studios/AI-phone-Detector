import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 to Core ML for Apple Neural Engine experiments.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path or name, for example yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size")
    args = parser.parse_args()

    model = YOLO(args.model)
    exported = model.export(format="coreml", imgsz=args.imgsz, nms=True)
    print(f"Exported Core ML model: {exported}")
    print("Note: live app inference still uses PyTorch MPS until a Core ML runtime path is integrated.")


if __name__ == "__main__":
    main()

from ultralytics import YOLO

def main():
    # Load pretrained weights
    model = YOLO("yolov8n.pt")

    # Train on CUDA:0
    model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,            # Optimal for 6GB VRAM
        device=0,            # Dedicated RTX 4050
        workers=2,
        name="roadpulse_v1",
        optimizer="AdamW",
        lr0=0.001,
        patience=15,         # Early stopping if mAP ceases to improve
        save=True
    )

if __name__ == "__main__":
    main()
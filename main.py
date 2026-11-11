import cv2
import json
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from src.gps_receiver import GPSFeed

def run_realtime_pipeline(video_source=0, model_path="models/best.pt", gps_port="COM3"):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_source)

    evidence_dir = Path("data/outputs/evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # Initialize live GPS receiver (auto-simulates if COM port not found)
    gps = GPSFeed(port=gps_port, simulate=False)
    gps.start()

    pothole_registry = {}
    frame_idx = 0

    print("[RoadPulse] System online. Monitoring road surface...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.40,
            iou=0.5,
            device=0,
            verbose=False
        )

        r = results[0]
        annotated_frame = r.plot()

        if r.boxes and r.boxes.id is not None:
            track_ids = r.boxes.id.int().cpu().tolist()
            boxes = r.boxes.xyxy.cpu().tolist()
            confs = r.boxes.conf.cpu().tolist()

            for tid, box, conf in zip(track_ids, boxes, confs):
                if tid not in pothole_registry:
                    w = box[2] - box[0]
                    h = box[3] - box[1]
                    area = w * h
                    severity = "Critical" if area > 18000 else ("Moderate" if area > 7000 else "Minor")

                    lat, lon = gps.get_coordinates()

                    # Crop and save evidence photo for governmental audits
                    x1, y1, x2, y2 = map(int, box)
                    pad = 20
                    crop_y1 = max(0, y1 - pad)
                    crop_y2 = min(frame.shape[0], y2 + pad)
                    crop_x1 = max(0, x1 - pad)
                    crop_x2 = min(frame.shape[1], x2 + pad)

                    crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    evidence_filename = f"defect_RPV_{tid:04d}.jpg"
                    evidence_path = evidence_dir / evidence_filename
                    cv2.imwrite(str(evidence_path), crop_img)

                    defect_entry = {
                        "defect_id": f"RPV-{tid:04d}",
                        "severity": severity,
                        "confidence": round(float(conf), 2),
                        "area_px": round(area, 1),
                        "latitude": lat,
                        "longitude": lon,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "evidence_path": str(evidence_path).replace("\\", "/"),
                        "status": "Pending Dispatch"
                    }
                    pothole_registry[tid] = defect_entry

                    # Continuously persist telemetry so dashboard updates live
                    with open("data/outputs/telemetry.json", "w") as f:
                        json.dump(list(pothole_registry.values()), f, indent=4)

        # Telemetry HUD
        lat, lon = gps.get_coordinates()
        cv2.rectangle(annotated_frame, (10, 10), (450, 95), (25, 25, 25), -1)
        cv2.putText(annotated_frame, f"RoadPulse | Live Pothole Monitor", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2)
        cv2.putText(annotated_frame, f"GPS: {lat:.5f} N, {lon:.5f} E", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"Hazards Logged: {len(pothole_registry)}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

        cv2.imshow("RoadPulse Real-Time Operator View", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    gps.stop()
    cap.release()
    cv2.destroyAllWindows()
    print(f"[Done] Pipeline concluded. Telemetry and cropped evidence saved.")

if __name__ == "__main__":
    # Point to test video, or use 0 for USB live dashcam
    run_realtime_pipeline("samples/test_road.mp4")
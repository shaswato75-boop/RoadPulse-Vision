# stream_server.py
import cv2
import json
import base64
import numpy as np
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from ultralytics import YOLO

app = FastAPI(title="RoadPulse Realtime Sentinel")

# Load YOLO model onto RTX 4050
model = YOLO("models/best.pt")

evidence_dir = Path("data/outputs/evidence")
evidence_dir.mkdir(parents=True, exist_ok=True)
telemetry_path = Path("data/outputs/telemetry.json")

# In-memory registry across frames
registry = {}

# Load existing telemetry if present
if telemetry_path.exists():
    try:
        with open(telemetry_path, "r") as f:
            for item in json.load(f):
                tid = int(item["defect_id"].replace("RPV-", ""))
                registry[tid] = item
    except Exception:
        pass

HTML_CLIENT = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>RoadPulse Sentinel</title>
    <style>
        body { margin: 0; background: #0f172a; color: #fff; font-family: -apple-system, sans-serif; text-align: center; }
        #video-container { position: relative; width: 100vw; height: 60vh; background: #000; overflow: hidden; }
        video { width: 100%; height: 100%; object-fit: cover; }
        #overlay-canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
        .hud { padding: 12px; display: flex; justify-content: space-around; background: #1e293b; font-size: 14px; }
        .hud div span { display: block; font-weight: bold; font-size: 18px; color: #38bdf8; }
        .controls { padding: 14px; }
        button { background: #ef4444; color: white; border: none; padding: 12px 24px; font-size: 16px; border-radius: 8px; font-weight: bold; width: 80%; }
        button.active { background: #22c55e; }
        #status { font-size: 12px; color: #94a3b8; margin-top: 6px; }
    </style>
</head>
<body>
    <div id="video-container">
        <video id="camera" autoplay playsinline muted></video>
        <canvas id="overlay-canvas"></canvas>
    </div>

    <div class="hud">
        <div>Total Detected<span id="stat-total">0</span></div>
        <div>Critical<span id="stat-critical" style="color: #f87171;">0</span></div>
        <div>FPS<span id="stat-fps">0</span></div>
    </div>

    <div class="controls">
        <button id="toggle-btn" onclick="toggleStream()">Start Live Patrol</button>
        <div id="status">Ready. Tap to start camera.</div>
    </div>

    <canvas id="capture-canvas" style="display:none;"></canvas>

    <script>
        const video = document.getElementById('camera');
        const overlay = document.getElementById('overlay-canvas');
        const captureCanvas = document.getElementById('capture-canvas');
        const toggleBtn = document.getElementById('toggle-btn');
        const statusText = document.getElementById('status');

        let ws = null;
        let streaming = false;
        let currentLat = 22.802500;
        let currentLon = 89.532800;
        let frameCount = 0;
        let lastFpsUpdate = Date.now();

        // High accuracy phone GPS tracking
        if (navigator.geolocation) {
            navigator.geolocation.watchPosition(
                (pos) => {
                    currentLat = pos.coords.latitude;
                    currentLon = pos.coords.longitude;
                },
                (err) => console.warn("GPS lock waiting..."),
                { enableHighAccuracy: true, maximumAge: 1000 }
            );
        }

        async function initCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: { ideal: "environment" }, width: { ideal: 640 }, height: { ideal: 480 } },
                    audio: false
                });
                video.srcObject = stream;
            } catch (err) {
                statusText.innerText = "Camera permission denied or not supported.";
            }
        }

        function toggleStream() {
            if (!streaming) {
                startWebSocket();
            } else {
                stopWebSocket();
            }
        }

        function startWebSocket() {
            const protocol = location.protocol === "https:" ? "wss:" : "ws:";
            ws = new WebSocket(`${protocol}//${location.host}/ws/stream`);

            ws.onopen = () => {
                streaming = true;
                toggleBtn.innerText = "Stop Patrol";
                toggleBtn.classList.add("active");
                statusText.innerText = "Streaming to RTX 4050 engine over Cloudflare...";
                sendFrames();
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                document.getElementById('stat-total').innerText = data.total_defects;
                document.getElementById('stat-critical').innerText = data.critical_count;

                frameCount++;
                const now = Date.now();
                if (now - lastFpsUpdate >= 1000) {
                    document.getElementById('stat-fps').innerText = frameCount;
                    frameCount = 0;
                    lastFpsUpdate = now;
                }

                // Draw bounding boxes on phone viewfinder
                drawDetections(data.detections);
            };

            ws.onclose = () => stopWebSocket();
            ws.onerror = () => stopWebSocket();
        }

        function stopWebSocket() {
            streaming = false;
            if (ws) ws.close();
            toggleBtn.innerText = "Start Live Patrol";
            toggleBtn.classList.remove("active");
            statusText.innerText = "Patrol stopped.";
            clearOverlay();
        }

        function sendFrames() {
            if (!streaming || !ws || ws.readyState !== WebSocket.OPEN) return;

            const ctx = captureCanvas.getContext('2d');
            captureCanvas.width = 480;
            captureCanvas.height = 360;
            ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

            const base64Data = captureCanvas.toDataURL('image/jpeg', 0.5);

            ws.send(JSON.stringify({
                image: base64Data,
                lat: currentLat,
                lon: currentLon
            }));

            // Control frame rate ~15-20 FPS for optimal cellular bandwidth
            setTimeout(sendFrames, 60);
        }

        function drawDetections(boxes) {
            overlay.width = video.clientWidth;
            overlay.height = video.clientHeight;
            const ctx = overlay.getContext('2d');
            ctx.clearRect(0, 0, overlay.width, overlay.height);

            const scaleX = overlay.width / 480;
            const scaleY = overlay.height / 360;

            boxes.forEach(b => {
                const x1 = b.box[0] * scaleX;
                const y1 = b.box[1] * scaleY;
                const w = (b.box[2] - b.box[0]) * scaleX;
                const h = (b.box[3] - b.box[1]) * scaleY;

                ctx.strokeStyle = b.severity === "Critical" ? "#ef4444" : "#f59e0b";
                ctx.lineWidth = 3;
                ctx.strokeRect(x1, y1, w, h);

                ctx.fillStyle = ctx.strokeStyle;
                ctx.font = "bold 13px sans-serif";
                ctx.fillText(`${b.defect_id} (${b.severity})`, x1, y1 > 16 ? y1 - 5 : y1 + 15);
            });
        }

        function clearOverlay() {
            const ctx = overlay.getContext('2d');
            ctx.clearRect(0, 0, overlay.width, overlay.height);
        }

        initCamera();
    </script>
</body>
</html>
"""


@app.get("/")
async def get_client():
    return HTMLResponse(HTML_CLIENT)


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw_msg = await websocket.receive_text()
            data = json.loads(raw_msg)

            # Extract phone frame and GPS
            encoded_image = data["image"].split(",")[1]
            jpg_bytes = base64.b64decode(encoded_image)
            np_arr = np.frombuffer(jpg_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            curr_lat = float(data.get("lat", 22.802500))
            curr_lon = float(data.get("lon", 89.532800))

            if frame is None:
                continue

            # Run YOLO + ByteTrack on GPU
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

            frame_detections = []

            if r.boxes and r.boxes.id is not None:
                track_ids = r.boxes.id.int().cpu().tolist()
                boxes = r.boxes.xyxy.cpu().tolist()
                confs = r.boxes.conf.cpu().tolist()

                for tid, box, conf in zip(track_ids, boxes, confs):
                    w = box[2] - box[0]
                    h = box[3] - box[1]
                    area = w * h
                    sev = "Critical" if area > 10000 else ("Moderate" if area > 4000 else "Minor")

                    frame_detections.append({
                        "defect_id": f"RPV-{tid:04d}",
                        "severity": sev,
                        "box": [round(c, 1) for c in box],
                        "conf": round(float(conf), 2)
                    })

                    # If new pothole, crop evidence and persist telemetry
                    if tid not in registry:
                        x1, y1, x2, y2 = map(int, box)
                        crop = frame[
                            max(0, y1 - 10):min(frame.shape[0], y2 + 10), max(0, x1 - 10):min(frame.shape[1], x2 + 10)]
                        evidence_file = evidence_dir / f"defect_RPV_{tid:04d}.jpg"
                        if crop.size > 0:
                            cv2.imwrite(str(evidence_file), crop)

                        registry[tid] = {
                            "defect_id": f"RPV-{tid:04d}",
                            "severity": sev,
                            "confidence": round(float(conf), 2),
                            "area_px": round(area, 1),
                            "latitude": round(curr_lat, 6),
                            "longitude": round(curr_lon, 6),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "evidence_path": str(evidence_file).replace("\\", "/")
                        }

                        # Save updated telemetry for dashboard
                        with open(telemetry_path, "w") as f:
                            json.dump(list(registry.values()), f, indent=4)

            # Send telemetry response back to phone
            critical_count = sum(1 for d in registry.values() if d["severity"] == "Critical")
            await websocket.send_text(json.dumps({
                "detections": frame_detections,
                "total_defects": len(registry),
                "critical_count": critical_count
            }))

    except WebSocketDisconnect:
        pass
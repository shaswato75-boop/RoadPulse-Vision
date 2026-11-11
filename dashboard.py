import streamlit as st
import cv2
import json
import urllib.parse
import pandas as pd
import folium
from streamlit_folium import st_folium
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ----------------- Configuration & Page Setup -----------------
st.set_page_config(
    page_title="RoadPulse-Vision | Municipal Civic Portal",
    page_icon="🛣️",
    layout="wide"
)

# Robust Path Setup
PROJECT_ROOT = Path(__file__).resolve().parent
upload_dir = PROJECT_ROOT / "data" / "uploads"
upload_dir.mkdir(parents=True, exist_ok=True)
evidence_dir = PROJECT_ROOT / "data" / "outputs" / "evidence"
evidence_dir.mkdir(parents=True, exist_ok=True)
telemetry_path = PROJECT_ROOT / "data" / "outputs" / "telemetry.json"
fallback_path = PROJECT_ROOT / "telemetry.json"

# Header Styling
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 95%;
        }
        .portal-header {
            text-align: center;
            margin-bottom: 25px;
        }
        .portal-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--text-color, #FAFAFA);
            margin-bottom: 6px;
        }
        .portal-subtitle {
            font-size: 1.05rem;
            color: var(--text-color, #A0AEC0);
            opacity: 0.85;
            margin-top: 0;
        }
    </style>
    <div class="portal-header">
        <h1 class="portal-title">🛣️ RoadPulse-Vision | Civic Defect Reporting & Action Portal</h1>
        <p class="portal-subtitle">Automated road hazard detection with instant authority dispatch and location reporting.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------- Sidebar: Mode & Input -----------------
st.sidebar.header("📡 Live Patrol Sync")
live_sync = st.sidebar.toggle("Auto-Refresh Telemetry", value=False, help="Enable this when monitoring live phone stream patrols over Cloudflare.")

# ONLY trigger autorefresh if the user explicitly enabled it and we are NOT processing an offline video
if live_sync and not st.session_state.get("is_processing", False):
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=3000, limit=None, key="live_patrol_autorefresh")

st.sidebar.markdown("---")
st.sidebar.header("📁 Offline Video Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload Dashcam Video",
    type=["mp4", "avi", "mov", "mkv"],
    help="Upload video files for batch inference."
)

saved_video_path = None
if uploaded_file is not None:
    saved_video_path = upload_dir / uploaded_file.name
    # Cache write to prevent rewriting on each run
    if not saved_video_path.exists() or saved_video_path.stat().st_size != uploaded_file.size:
        with open(saved_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"Ready: `{uploaded_file.name}`")

run_analysis = st.sidebar.button(
    "🚀 Run Offline Video Analysis",
    type="primary",
    disabled=(uploaded_file is None),
    use_container_width=True
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Survey Parameters")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.20, 0.85, 0.40, 0.05)
start_lat = st.sidebar.number_input("Start Latitude", value=22.802500, format="%.6f")
start_lon = st.sidebar.number_input("Start Longitude", value=89.532800, format="%.6f")

st.sidebar.subheader("🏛️ Authority Contact Details")
authority_email = st.sidebar.text_input("Authority Email", value="complaints@rhd.gov.bd")
authority_phone = st.sidebar.text_input("Control Room WhatsApp", value="8801700000000")

# ----------------- Offline Video Processing Engine -----------------
if run_analysis and saved_video_path is not None:
    st.session_state["is_processing"] = True

    st.info("⚙️ Processing video on RTX 4050 GPU...")
    cap = cv2.VideoCapture(str(saved_video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    try:
        model = YOLO("models/best.pt")
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        st.session_state["is_processing"] = False
        st.stop()

    registry = {}
    curr_lat, curr_lon = start_lat, start_lon
    frame_idx = 0

    col_stream, col_kpi = st.columns([3, 1])
    stream_placeholder = col_stream.empty()
    kpi_placeholder = col_kpi.empty()
    prog_bar = st.progress(0.0)

    # Isolated processing loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf_thresh,
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
                if tid not in registry:
                    w, h = box[2] - box[0], box[3] - box[1]
                    area = w * h
                    sev = "Critical" if area > 18000 else ("Moderate" if area > 7000 else "Minor")

                    curr_lat += 0.00008
                    curr_lon += 0.00005

                    x1, y1, x2, y2 = map(int, box)
                    crop_img = frame[
                        max(0, y1 - 15):min(frame.shape[0], y2 + 15),
                        max(0, x1 - 15):min(frame.shape[1], x2 + 15)
                    ]
                    evidence_file = evidence_dir / f"defect_RPV_{tid:04d}.jpg"
                    if crop_img.size > 0:
                        cv2.imwrite(str(evidence_file), crop_img)

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

        # Update preview every 3 frames to keep UI responsive
        if frame_idx % 3 == 0:
            stream_placeholder.image(
                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True
            )

        prog = min(frame_idx / total_frames, 1.0)
        prog_bar.progress(prog)
        kpi_placeholder.markdown(f"### Processing Video\n* **Defects:** `{len(registry)}`\n* **Frame:** `{frame_idx}/{total_frames}`")

    cap.release()
    prog_bar.empty()

    # Save to disk
    with open(telemetry_path, "w", encoding="utf-8") as f:
        json.dump(list(registry.values()), f, indent=4)

    st.session_state["is_processing"] = False
    st.success("✅ Analysis complete! Dashboard updated below.")

# ----------------- Unified Telemetry Loader -----------------
records = []
target_file = None

if telemetry_path.exists():
    target_file = telemetry_path
elif fallback_path.exists():
    target_file = fallback_path

if target_file and target_file.is_file():
    try:
        with open(target_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                records = json.loads(content)
    except Exception as e:
        st.sidebar.error(f"Error reading telemetry: {e}")

# Sidebar Diagnostic Status
st.sidebar.markdown("---")
st.sidebar.subheader("📡 Telemetry Monitor")
st.sidebar.write(f"**Source:** `{target_file.name if target_file else 'None'}`")
st.sidebar.write(f"**Defects Logged:** `{len(records)}`")

if st.sidebar.button("🗑️ Clear Telemetry Log", use_container_width=True):
    if target_file and target_file.exists():
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump([], f)
    st.rerun()

# ----------------- Dashboard Visualizer & Dispatch -----------------
if records and len(records) > 0:
    df = pd.DataFrame(records)
    df["severity"] = df["severity"].astype(str).str.capitalize()

    crit_count = len(df[df["severity"] == "Critical"])
    mod_count = len(df[df["severity"] == "Moderate"])
    min_count = len(df[df["severity"] == "Minor"])
    avg_lat = df["latitude"].mean()
    avg_lon = df["longitude"].mean()

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Defects Logged", len(df))
    c2.metric("Critical Hazards", crit_count)
    c3.metric("Moderate Repairs", mod_count)
    c4.metric("Minor Flaws", min_count)

    st.divider()
    map_col, action_col = st.columns([3, 2])

    COLOR_MAP = {"Critical": "#D32F2F", "Moderate": "#F57C00", "Minor": "#1976D2"}

    with map_col:
        st.subheader("📍 Geospatial Defect Distribution")
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=15, tiles="OpenStreetMap")
        for _, row in df.iterrows():
            sev = row["severity"]
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=8 if sev == "Critical" else (6 if sev == "Moderate" else 4),
                popup=f"<b>{row['defect_id']}</b> ({sev})<br>Lat: {row['latitude']:.5f}<br>Lon: {row['longitude']:.5f}",
                color=COLOR_MAP.get(sev, "#757575"),
                fill=True,
                fill_opacity=0.85
            ).add_to(m)
        st_folium(m, width="100%", height=460)

    with action_col:
        st.subheader("🔍 Single Defect Inspection")
        selected_id = st.selectbox("Select Defect Record", df["defect_id"].tolist())
        s_row = df[df["defect_id"] == selected_id].iloc[0]

        evidence_str = str(s_row.get("evidence_path", "")).strip()
        evidence_file = Path(evidence_str) if evidence_str else None

        if evidence_file and evidence_file.is_file():
            st.image(
                str(evidence_file),
                caption=f"{selected_id} | Severity: {s_row['severity']}",
                use_container_width=True
            )
        else:
            st.warning(f"📸 No snapshot file found on disk for {selected_id}")

        pin_url = f"https://www.google.com/maps?q={s_row['latitude']},{s_row['longitude']}"
        single_text = f"Defect ID: {selected_id}\nSeverity: {s_row['severity']}\nGPS: {s_row['latitude']}, {s_row['longitude']}\nMap Pin: {pin_url}"
        st.text_area("Single Hazard Notice", single_text, height=105)

    # ----------------- Authority Dispatch Actions -----------------
    st.divider()
    st.subheader("📢 Send All Locations to Authority (Complete Report)")
    report_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_dossier_lines = [
        "=" * 60,
        "OFFICIAL ROAD INFRASTRUCTURE DEFECT REPORT",
        "=" * 60,
        f"Generated: {report_timestamp}",
        f"Corridor Overview Pin: https://www.google.com/maps?q={avg_lat:.5f},{avg_lon:.5f}",
        f"Summary: Total {len(df)} Defects (Critical: {crit_count}, Moderate: {mod_count}, Minor: {min_count})",
        "-" * 60,
        "COMPLETE ITEMIZED DEFECT INVENTORY (ALL LOCATIONS):"
    ]

    for idx, r in df.iterrows():
        map_link = f"https://www.google.com/maps?q={r['latitude']},{r['longitude']}"
        full_dossier_lines.append(
            f"{idx + 1:02d}. [{r['defect_id']}] {r['severity'].upper()} | GPS: {r['latitude']:.5f}, {r['longitude']:.5f} | Pin: {map_link}"
        )

    full_dossier_text = "\n".join(full_dossier_lines)

    with st.expander("📋 Preview Complete Audit Dossier Text", expanded=False):
        st.text_area("Full Report", full_dossier_text, height=220)

    crit_sample = df[df["severity"] == "Critical"].head(8)
    compact_pins = [
        f"- {r['defect_id']} (CRITICAL): https://www.google.com/maps?q={r['latitude']},{r['longitude']}"
        for _, r in crit_sample.iterrows()
    ]

    email_summary = (
        f"OFFICIAL ROAD DEFECT AUDIT - {len(df)} HAZARDS LOGGED\n"
        f"Date: {report_timestamp}\n"
        f"Critical: {crit_count} | Moderate: {mod_count} | Minor: {min_count}\n"
        f"Corridor Overview: https://www.google.com/maps?q={avg_lat:.5f},{avg_lon:.5f}\n\n"
        f"IMMEDIATE ATTENTION REQUIRED (TOP CRITICAL PINS):\n" + "\n".join(compact_pins) +
        f"\n\n[Full audit file with all {len(df)} itemized defects has been exported.]"
    )

    encoded_subj = urllib.parse.quote(f"URGENT: Road Condition Report - {len(df)} Defects ({crit_count} Critical)")
    encoded_body = urllib.parse.quote(email_summary)
    mailto_url = f"mailto:{authority_email}?subject={encoded_subj}&body={encoded_body}"
    wa_url = f"https://wa.me/{authority_phone}?text={encoded_body}"

    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown(
            f'<a href="{mailto_url}" target="_blank" style="display:block;text-align:center;background-color:#B71C1C;color:white;padding:12px;border-radius:6px;text-decoration:none;font-weight:bold;">📧 Email Summary to Authority</a>',
            unsafe_allow_html=True
        )
    with b2:
        st.markdown(
            f'<a href="{wa_url}" target="_blank" style="display:block;text-align:center;background-color:#1B5E20;color:white;padding:12px;border-radius:6px;text-decoration:none;font-weight:bold;">💬 WhatsApp Summary</a>',
            unsafe_allow_html=True
        )
    with b3:
        st.download_button(
            label="📄 Download Complete Report (.txt)",
            data=full_dossier_text,
            file_name=f"roadpulse_complete_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.divider()
    dl1, dl2 = st.columns(2)
    dl1.download_button(
        label="📊 Download Complete CSV Telemetry",
        data=df.to_csv(index=False),
        file_name="roadpulse_all_defects.csv",
        mime="text/csv",
        use_container_width=True
    )
    dl2.download_button(
        label="💾 Download Complete JSON Telemetry",
        data=json.dumps(records, indent=4),
        file_name="roadpulse_all_defects.json",
        mime="application/json",
        use_container_width=True
    )
else:
    st.info("ℹ️ No defect telemetry recorded yet. Either toggle **Auto-Refresh Telemetry** during live phone streaming, or upload a video and click **🚀 Run Offline Video Analysis**.")
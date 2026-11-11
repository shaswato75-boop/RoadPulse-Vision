# src/gps_sync.py
import re
from datetime import datetime


def parse_dashcam_srt(srt_path):
    """
    Parses standard dashcam subtitle tracks (.srt) that store
    GPS coords like: [lat: 22.845123] [lon: 89.541201] [speed: 42 km/h]
    """
    gps_timeline = []
    coord_pattern = re.compile(r'([+-]?\d+\.\d+)\s*,\s*([+-]?\d+\.\d+)')

    with open(srt_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    entries = content.split('\n\n')
    for entry in entries:
        lines = entry.strip().split('\n')
        if len(lines) >= 3:
            # line 1: timestamp range (00:00:01,000 --> 00:00:02,000)
            # line 2+: text with coords
            match = coord_pattern.search(lines[2])
            if match:
                lat, lon = float(match.group(1)), float(match.group(2))
                gps_timeline.append((lat, lon))
    return gps_timeline
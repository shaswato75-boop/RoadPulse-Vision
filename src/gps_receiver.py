import threading
import time

try:
    import serial
    import pynmea2
except ImportError:
    serial = None
    pynmea2 = None

class GPSFeed:
    def __init__(self, port="COM3", baudrate=9600, simulate=False):
        self.port = port
        self.baudrate = baudrate
        self.simulate = simulate
        self.latitude = 22.802500  # Default initial coordinate (Khulna area)
        self.longitude = 89.532800
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        if not self.simulate and serial and pynmea2:
            try:
                ser = serial.Serial(self.port, baudrate=self.baudrate, timeout=1)
                while self.running:
                    line = ser.readline().decode('ascii', errors='replace').strip()
                    if line.startswith('$GPGGA') or line.startswith('$GNGGA'):
                        try:
                            msg = pynmea2.parse(line)
                            if msg.latitude and msg.longitude:
                                self.latitude = msg.latitude
                                self.longitude = msg.longitude
                        except Exception:
                            continue
            except Exception as e:
                print(f"[GPS] Serial unavailable ({e}). Falling back to simulation mode.")
                self.simulate = True

        # Fallback / simulated vehicle movement
        while self.running:
            self.latitude += 0.00003
            self.longitude += 0.00002
            time.sleep(0.5)

    def get_coordinates(self):
        return round(self.latitude, 6), round(self.longitude, 6)

    def stop(self):
        self.running = False
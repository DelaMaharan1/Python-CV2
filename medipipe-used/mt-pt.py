import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np

# Hard core code hand marker from git 
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),    # Jempol
    (0, 5), (5, 6), (6, 7), (7, 8),    # Telunjuk
    (0, 9), (9, 10), (10, 11), (11, 12), # Tengah
    (0, 13), (13, 14), (14, 15), (15, 16), # Manis
    (0, 17), (17, 18), (18, 19), (19, 20), # Kelingking
    (5, 9), (9, 13), (13, 17)              # Telapak
]

# Variabel Global untuk menyimpan hasil deteksi
latest_result = None

def to_pixel(x_norm, y_norm, w, h):
    x = min(max(x_norm, 0.0), 1.0)
    y = min(max(y_norm, 0.0), 1.0)
    return int(x * w), int(y * h)

def print_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

# Setup Options
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=print_result
)

cap = cv2.VideoCapture(1) 

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1) # Mirror
        h, w, _ = frame.shape
        
        # Proses MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

        # output landmark results
        if latest_result is not None and latest_result.hand_landmarks:
            for hand_landmarks in latest_result.hand_landmarks:
                # Ambil koordinat pixel
                pts = [to_pixel(lm.x, lm.y, w, h) for lm in hand_landmarks]

                # Gambar Garis (Connections)
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2) # Warna Hijau

                # Gambar Titik (Points)
                for (x, y) in pts:
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1) # Warna Merah

        cv2.imshow("Manual Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
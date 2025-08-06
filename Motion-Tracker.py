import cv2
import numpy as np

cap = cv2.VideoCapture('Cars driving at night.mp4')

# Pastikan kamera terbuka
if not cap.isOpened():
    print("Tidak bisa membuka kamera.")
    exit()

ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    if frame1 is None or frame2 is None:
        print("Frame tidak valid. Menghentikan proses.")
        break

    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    movement_detected = False

    for contour in contours:
        if cv2.contourArea(contour) < 700:
            continue
        movement_detected = True
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 1)  # lebih tipis

    if movement_detected:
        cv2.putText(frame1, "Status: Movement", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("feed", frame1)

    frame1 = frame2
    ret, frame2 = cap.read()
    
    if not ret:
        print("Tidak bisa membaca frame berikutnya.")
        break

    if cv2.waitKey(30) == 27:  # ESC untuk keluar
        break

cap.release()
cv2.destroyAllWindows()


import cv2
import numpy as np

cap = cv2.VideoCapture('Cars driving at night.mp4')
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 700:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Tambahkan teks status sekali per frame (kalau ada gerakan terdeteksi)
    if any(cv2.contourArea(c) >= 700 for c in contours):
        cv2.putText(frame1, "Status: Movement", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("feed", frame1)

    frame1 = frame2
    ret, frame2 = cap.read()

    if not ret:
        break

    if cv2.waitKey(40) == 27:  # tekan ESC untuk keluar
        break

cap.release()
cv2.destroyAllWindows()

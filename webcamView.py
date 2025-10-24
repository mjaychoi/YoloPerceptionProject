from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model(frame, stream=False, conf=0.4)
    annotated = results[0].plot()   # draw boxes on frame

    cv2.imshow("YOLOv8 Live", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

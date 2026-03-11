import cv2
from ultralytics import YOLO

# load your trained model
model = YOLO("runs/detect/train4/weights/best.pt")

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("Fruit Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:   # press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
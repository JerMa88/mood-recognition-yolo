from ultralytics import YOLO
import cv2
import torch

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load model
model = YOLO('../../data/saved_models/yolov8n-cls-best_1.pt')

# Object classes
classNames = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

while True:
    success, img = cap.read()
    if not success:
        break

    # Run classification
    results = model(img)

    # Get top prediction
    top_result = results[0]
    cls = int(top_result.probs.top1)
    confidence = top_result.probs.top1conf.item()

    # Set color based on class
    if classNames[cls] == "happy":
        color = (0, 128, 0)  # Green
    else:
        color = (173, 216, 230)  # Light blue

    # Display class name and confidence
    label = f"{classNames[cls]} {confidence:.2f}"
    cv2.putText(img, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow('Webcam Classification', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
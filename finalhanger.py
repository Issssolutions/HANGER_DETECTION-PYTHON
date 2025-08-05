import numpy as np
from ultralytics import YOLO
import cv2
import cvzone

# Load video or camera
cap = cv2.VideoCapture('videos/11.mp4')
# cap = cv2.VideoCapture(0)  # For webcam

# Load custom-trained YOLO model
model = YOLO("yolo_weights/lastlast.pt")  # Replace with your actual model path

# Set your class name
classNames = ['Hanger']

# Confidence threshold
confidence_threshold = 0.6

# Tracking setup
previous_centers = {}
hanger_count = 0
counting_line_x = 650
id_counter = 0

# Get video properties
frame_width = 1300
frame_height = 800
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Set up VideoWriter to save output
output_path = "output/detected_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    # Preprocess the frame
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (frame_width, frame_height))
    imgHeight, imgWidth = img.shape[:2]

    # YOLO prediction
    results = model(img, stream=True)
    new_centers = {}

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            conf = round(box.conf[0].item(), 2)
            cls = int(box.cls[0])

            if cls < len(classNames) and classNames[cls] == 'Hanger' and conf >= confidence_threshold:
                cx = x1 + w // 2
                cy = y1 + h // 2

                id_counter += 1
                object_id = id_counter
                new_centers[object_id] = (cx, cy)

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                label = f"{classNames[cls]} {conf}"
                cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=5)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9)

    for id, (cx, cy) in new_centers.items():
        if id - 1 in previous_centers:
            prev_cx, _ = previous_centers[id - 1]
            if prev_cx < counting_line_x and cx >= counting_line_x:
                hanger_count += 1

    previous_centers = new_centers.copy()

    # Draw line and counter
    cv2.line(img, (counting_line_x, 0), (counting_line_x, imgHeight), (0, 255, 0), 2)
    cvzone.putTextRect(
        img, f'Hanger Count: {hanger_count}', (20, 50),
        scale=2, thickness=2, offset=5, colorR=(0, 255, 0)
    )

    # Write frame to output video
    out.write(img)

    # Show frame (optional)
    cv2.imshow("Hanger Detection & Counting", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import threading
import queue
from ultralytics import YOLO
from sort import Sort
import cvzone

# Load YOLO model
model = YOLO("yolo_weights/lastlast.pt")
classNames = ["Hanger"]
confidence_threshold = 0.6

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Resize dimensions
FRAME_WIDTH = 1300
FRAME_HEIGHT = 800

# ROI line
line_x = 650
offset = 15
counted_ids = set()
id_positions = {}
total_count = 0

# Queues
frame_queue = queue.Queue(maxsize=5)
result_queue = queue.Queue(maxsize=5)

# Capture setup
cap = cv2.VideoCapture("videos/11.mp4")


def frame_reader():
    while True:
        ret, frame = cap.read()
        if not ret:
            frame_queue.put(None)
            break
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame_queue.put(frame)


def model_inferencer():
    while True:
        frame = frame_queue.get()
        if frame is None:
            result_queue.put(None)
            break

        results = model(frame, stream=False)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()
            if conf > confidence_threshold:
                detections.append([x1, y1, x2, y2, conf])

        result_queue.put((frame, detections))


# Start threads
reader_thread = threading.Thread(target=frame_reader)
inferencer_thread = threading.Thread(target=model_inferencer)
reader_thread.start()
inferencer_thread.start()

# Main loop (processing + display)
while True:
    item = result_queue.get()
    if item is None:
        break

    frame, detections = item

    # Draw detections
    for det in detections:
        x1, y1, x2, y2, conf = det
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame, (int(x1), int(y1), int(w), int(h)), l=9, rt=2)

    # Tracker
    detections_np = np.array(detections) if detections else np.empty((0, 5))
    track_results = tracker.update(detections_np)

    # Draw vertical line
    cv2.line(frame, (line_x, 0), (line_x, FRAME_HEIGHT), (255, 0, 255), 2)

    # Tracking logic
    for track in track_results:
        x1, y1, x2, y2, track_id = map(int, track)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Draw tracking info
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Count logic
        if track_id not in id_positions:
            id_positions[track_id] = cx
        else:
            prev_cx = id_positions[track_id]
            if prev_cx < line_x - offset and cx >= line_x + offset:
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                    total_count += 1
                    print(f"Hanger Counted! ID={track_id} | Total={total_count}")
            id_positions[track_id] = cx

    # Show count
    cv2.putText(frame, f'Total Count: {total_count}', (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    # Display
    cv2.imshow("Hanger Detection & Counting", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
reader_thread.join()
inferencer_thread.join()

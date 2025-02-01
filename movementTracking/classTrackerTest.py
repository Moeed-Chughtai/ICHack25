import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import csv
import os

# -----------------------------
# Initialize Models and Utilities
# -----------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model', 'yolov8n.pt')
video_path = os.path.join(base_dir, 'videos', 'school_kid_class1.mp4')
csv_filename = os.path.join(base_dir, 'logs', 'real_time_motion_log.csv')

# Load YOLO model and tracker
yolo_model = YOLO(model_path)
tracker = DeepSort(max_age=30, n_init=5)

cap = cv2.VideoCapture(video_path)

NUM_STUDENTS = 5  # Adjust as needed
movement_dict = {}

fps = cap.get(cv2.CAP_PROP_FPS) or 30

frame_count = 0

# Function to calculate IoU for bounding box matching
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

IOU_THRESHOLD = 0.5

# Ensure the logs directory exists
os.makedirs(os.path.join(base_dir, 'logs'), exist_ok=True)

# Initialize CSV file for real-time logging
with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=['frame', 'student_id', 'movement_delta', 'cumulative_movement'])
    writer.writeheader()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_resized = cv2.resize(frame, (640, 480))

            # YOLO detection
            results = yolo_model(frame_resized)[0]

            detections = []
            for result in results.boxes.data:
                x1, y1, x2, y2, conf, cls = result
                if int(cls) == 0 and conf > 0.5:
                    bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                    detections.append((bbox, conf, None))

            detections = detections[:NUM_STUDENTS]
            tracks = tracker.update_tracks(detections, frame=frame_resized)

            for track in tracks:
                if not track.is_confirmed() or track.hits < 3:
                    continue

                track_id = track.track_id
                if len(movement_dict) >= NUM_STUDENTS and track_id not in movement_dict:
                    continue

                l, t, r, b = track.to_ltrb()
                current_position = np.array([(l + r) / 2, (t + b) / 2])
                current_bbox = [l, t, r, b]

                matched_id = None
                for existing_id, data in movement_dict.items():
                    if calculate_iou(current_bbox, data['bbox']) > IOU_THRESHOLD:
                        matched_id = existing_id
                        break

                if matched_id is None:
                    matched_id = track_id
                    movement_dict[matched_id] = {
                        'prev_position': current_position,
                        'cumulative_movement': 0,
                        'bbox': current_bbox
                    }
                    movement_delta = 0
                else:
                    prev_position = movement_dict[matched_id]['prev_position']
                    movement_delta = np.linalg.norm(current_position - prev_position)
                    movement_dict[matched_id]['cumulative_movement'] += movement_delta
                    movement_dict[matched_id]['prev_position'] = current_position
                    movement_dict[matched_id]['bbox'] = current_bbox

                writer.writerow({
                    'frame': frame_count,
                    'student_id': matched_id,
                    'movement_delta': movement_delta,
                    'cumulative_movement': movement_dict[matched_id]['cumulative_movement']
                })

                cv2.putText(frame_resized, f"Student {matched_id}: {movement_dict[matched_id]['cumulative_movement']:.2f}", 
                            (int(l), int(t) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame_resized, (int(l), int(t)), (int(r), int(b)), (255, 0, 0), 2)

            cv2.imshow('Real-Time Movement Tracking', frame_resized)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

print(f"Motion data logged in real-time to {csv_filename}")

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from deepface import DeepFace
import csv
import os
from collections import defaultdict

# -----------------------------
# Configuration
# -----------------------------
# Set this to True to enable the frontal face check.
# Set to False to disable it (i.e. run emotion detection on all faces).
USE_FRONTAL_FACE_CHECK = True

# Maximum number of students to track.
NUM_STUDENTS = 25

# -----------------------------
# Initialize Models and Utilities
# -----------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model', 'yolov8n.pt')
video_path = os.path.join(base_dir, 'videos', 'zoom.mp4')
csv_filename = os.path.join(base_dir, 'logs', 'real_time_emotion_log.csv')

# Load YOLO model and DeepSort tracker
# Using n_init=1 to confirm new tracks immediately.
yolo_model = YOLO(model_path)
tracker = DeepSort(max_age=30, n_init=1)

cap = cv2.VideoCapture(video_path)

# This dictionary maps a student ID to its last known bounding box [l, t, r, b]
tracked_students = {}  

# This dictionary accumulates emotion durations per student.
emotion_dict = defaultdict(lambda: defaultdict(float))

fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_count = 0

# Calculate maximum number of frames for 10 seconds.
max_frames = int(fps * 30)

# Ensure the logs directory exists
os.makedirs(os.path.join(base_dir, 'logs'), exist_ok=True)

# Create the outputVideos directory if it does not exist.
output_videos_dir = os.path.join(base_dir, 'outputVideos')
os.makedirs(output_videos_dir, exist_ok=True)

# Initialize the VideoWriter to save the output video.
# Since we are resizing frames to 640x480, we use that as the frame size.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = os.path.join(output_videos_dir, 'output_video.mp4')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (640, 480))

# Function to calculate IoU for bounding box matching.
# Boxes are expected in [x1, y1, x2, y2] format.
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

# Function to check if one box is completely contained within another.
# Boxes are in [x1, y1, x2, y2] format.
def is_contained(inner_box, outer_box):
    return (inner_box[0] >= outer_box[0] and 
            inner_box[1] >= outer_box[1] and 
            inner_box[2] <= outer_box[2] and 
            inner_box[3] <= outer_box[3])

# Face orientation detection (ensures frontal faces)
def is_face_frontal(face_roi):
    """
    Using adjusted parameters for a more lenient detection.
    scaleFactor=1.1 and minNeighbors=1 means the cascade is less strict.
    """
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
    # For debugging, you can uncomment the next line:
    # print(f"Detected {len(faces)} face(s) in ROI")
    return len(faces) > 0

# Overlap parameter: if two boxes overlap more than this threshold (IoU > 0.5),
# we consider them redundant.
IOU_THRESHOLD = 0.5

# Initialize CSV file for real-time logging
with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=['frame', 'student_id', 'emotion', 'emotion_duration'])
    writer.writeheader()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Stop processing after the first 10 seconds.
            if frame_count > max_frames:
                break

            # (Optional) Pre-process frame here (brightness/contrast adjustments) if needed.
            frame_resized = cv2.resize(frame, (640, 480))
            h, w, _ = frame_resized.shape

            # YOLO detection
            results = yolo_model(frame_resized)[0]

            # Build the detections list while filtering out overlapping boxes.
            detections = []
            for result in results.boxes.data:
                x1, y1, x2, y2, conf, cls = result
                # Only select persons (class 0) with confidence > 0.1
                if int(cls) == 0 and conf > 0.15:
                    # Convert from (x1, y1, x2, y2) to [x, y, width, height]
                    bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                    # Convert to [x1, y1, x2, y2] for overlap calculations.
                    new_box = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                    
                    skip_box = False
                    indices_to_remove = []
                    for i, (existing_bbox, existing_conf, _) in enumerate(detections):
                        existing_box = [existing_bbox[0],
                                        existing_bbox[1],
                                        existing_bbox[0] + existing_bbox[2],
                                        existing_bbox[1] + existing_bbox[3]]
                        # Check if the new box overlaps more than threshold or one is contained in the other.
                        if calculate_iou(new_box, existing_box) > IOU_THRESHOLD or \
                           is_contained(new_box, existing_box) or is_contained(existing_box, new_box):
                            # Keep the one with higher confidence.
                            if conf > existing_conf:
                                indices_to_remove.append(i)
                            else:
                                skip_box = True
                                break
                    # Remove lower confidence detections (if any).
                    for idx in sorted(indices_to_remove, reverse=True):
                        detections.pop(idx)
                    if not skip_box:
                        detections.append((bbox, float(conf), None))

            # Update tracks with non-overlapping detections.
            tracks = tracker.update_tracks(detections, frame=frame_resized)

            for track in tracks:
                # Get the track's bounding box in l,t,r,b format.
                l, t, r, b = track.to_ltrb()
                l = max(0, int(l))
                t = max(0, int(t))
                r = min(w, int(r))
                b = min(h, int(b))
                new_box = [l, t, r, b]  # current bounding box for this track

                # Determine the student ID to assign.
                assigned_id = None
                if track.track_id in tracked_students:
                    # Existing student; update their bounding box.
                    assigned_id = track.track_id
                    tracked_students[assigned_id] = new_box
                else:
                    # For a new track, check if its box overlaps with an existing student.
                    for student_id, box in tracked_students.items():
                        if calculate_iou(new_box, box) > IOU_THRESHOLD or \
                           is_contained(new_box, box) or is_contained(box, new_box):
                            assigned_id = student_id
                            # Update the student's bounding box.
                            tracked_students[student_id] = new_box
                            break
                    if assigned_id is None:
                        # If no overlap is found, only add as a new student if we haven't reached the limit.
                        if len(tracked_students) < NUM_STUDENTS:
                            tracked_students[track.track_id] = new_box
                            assigned_id = track.track_id
                        else:
                            # Maximum number of students reached; ignore this new track.
                            continue

                # Optional: expand the bounding box slightly for the face region.
                pad = 10
                l_padded = max(l - pad, 0)
                t_padded = max(t - pad, 0)
                r_padded = min(r + pad, w)
                b_padded = min(b + pad, h)
                face_roi = frame_resized[t_padded:b_padded, l_padded:r_padded]
                if face_roi.size == 0:
                    continue

                # Determine the dominant emotion.
                if USE_FRONTAL_FACE_CHECK:
                    if is_face_frontal(face_roi):
                        try:
                            analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                            if isinstance(analysis, list):
                                analysis = analysis[0]
                            dominant_emotion = analysis['dominant_emotion']
                            emotion_confidence = analysis['emotion'][dominant_emotion]
                            if emotion_confidence < 0.7:
                                dominant_emotion = 'unknown'
                        except Exception as e:
                            dominant_emotion = 'unknown'
                    else:
                        dominant_emotion = 'not frontal'
                else:
                    try:
                        analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                        if isinstance(analysis, list):
                            analysis = analysis[0]
                        dominant_emotion = analysis['dominant_emotion']
                        emotion_confidence = analysis['emotion'][dominant_emotion]
                        if emotion_confidence < 0.7:
                            dominant_emotion = 'unknown'
                    except Exception as e:
                        dominant_emotion = 'unknown'

                # Update emotion duration for this student.
                emotion_dict[assigned_id][dominant_emotion] += 1 / fps

                # Write the log data to CSV.
                writer.writerow({
                    'frame': frame_count,
                    'student_id': assigned_id,
                    'emotion': dominant_emotion,
                    'emotion_duration': emotion_dict[assigned_id][dominant_emotion]
                })

                # Display the information on the frame.
                cv2.putText(frame_resized, f"ID {assigned_id} | {dominant_emotion}",
                            (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame_resized, (l, t), (r, b), (255, 0, 0), 2)

            # Write the processed frame to the output video.
            video_writer.write(frame_resized)

            # Display the frame.
            cv2.imshow('Real-Time Emotion Tracking', frame_resized)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        video_writer.release()  # Release the VideoWriter
        cv2.destroyAllWindows()

# Summary Report
print(f"Emotion data logged in real-time to {csv_filename}")
print(f"Output video (first 10 seconds) saved to {output_video_path}")
for student_id, emotions in emotion_dict.items():
    print(f"\nStudent {student_id} Emotion Summary:")
    for emotion, duration in emotions.items():
        print(f" - {emotion}: {duration:.2f} seconds")

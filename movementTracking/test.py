import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from deepface import DeepFace
import csv
import os
from collections import defaultdict

# -----------------------------
# Initialize Models and Utilities
# -----------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model', 'yolov8n.pt')
video_path = os.path.join(base_dir, 'videos', 'zoom.mp4')
csv_filename = os.path.join(base_dir, 'logs', 'real_time_emotion_log.csv')
student_images_dir = os.path.join(base_dir, 'images') # Directory containing student images
student_name_dict = {} # Dictionary to map track_id to student name

# Load student images for face recognition
if not os.path.exists(student_images_dir):
    os.makedirs(student_images_dir)
    print(f"Please put student images (name.jpg) in '{student_images_dir}'")
else:
    for filename in os.listdir(student_images_dir):
        if filename.endswith((".jpg", ".jpeg", ".png")): # Check for image extensions only
            try:
                student_name = filename.split('.')[0] # Extract name from filename (e.g., name.jpg -> name)
                student_name_dict[student_name] = filename # Use student name as key
            except ValueError:
                print(f"Warning: Invalid student image filename: {filename}. Expected format: name.jpg")

face_recognition_enabled = bool(student_name_dict) # Enable face recognition if student images are loaded
print(f"Face recognition enabled: {face_recognition_enabled}")
# Load YOLO model and DeepSort tracker
yolo_model = YOLO(model_path)
tracker = DeepSort(max_age=30, n_init=5)

cap = cv2.VideoCapture(video_path)

NUM_STUDENTS = 5
movement_dict = {}  # To store unique track IDs
emotion_dict = defaultdict(lambda: defaultdict(float))

fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_count = 0

# Ensure the logs directory exists
os.makedirs(os.path.join(base_dir, 'logs'), exist_ok=True)

# Function to calculate IoU for bounding box matching (not used in this fix)
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

# Face orientation detection (ensures frontal faces)
def is_face_frontal(face_roi):
    """
    Using adjusted parameters for a more lenient detection.
    scaleFactor=1.5 and minNeighbors=2 means the cascade is less strict,
    which can detect more faces even if they are not perfectly frontal.
    """
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=2)
    # Uncomment the following line for debugging:
    # print(f"Detected {len(faces)} face(s) in ROI")
    return len(faces) > 0

# IOU threshold (not used in this fix)
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
            # Resize frame for processing; adjust size if needed.
            frame_resized = cv2.resize(frame, (640, 480))
            h, w, _ = frame_resized.shape

            # YOLO detection
            results = yolo_model(frame_resized)[0]

            detections = []
            for result in results.boxes.data:
                x1, y1, x2, y2, conf, cls = result
                # Only select persons (class 0) with a confidence > 0.5
                if int(cls) == 0 and conf > 0.5:
                    # Convert from (x1, y1, x2, y2) to (x, y, width, height)
                    bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                    detections.append((bbox, float(conf), None))

            # Do not limit the detections (allow all detected persons)
            # detections = detections[:NUM_STUDENTS]  <-- REMOVED

            # Update tracks with all detections
            tracks = tracker.update_tracks(detections, frame=frame_resized)

            for track in tracks:
                # Process only confirmed tracks
                if not track.is_confirmed():
                    continue

                track_id = track.track_id

                # Keep track of unique student IDs, up to NUM_STUDENTS.
                if track_id not in movement_dict:
                    if len(movement_dict) < NUM_STUDENTS:
                        movement_dict[track_id] = True
                    else:
                        # Skip new tracks if we've already reached the limit
                        continue

                # Get the bounding box in l, t, r, b format
                l, t, r, b = track.to_ltrb()

                # Boundary check and ensure integers
                l = max(0, int(l))
                t = max(0, int(t))
                r = min(w, int(r))
                b = min(h, int(b))

                # Optional: expand the bounding box slightly to capture more of the face.
                pad = 10  # pixels
                l_padded = max(l - pad, 0)
                t_padded = max(t - pad, 0)
                r_padded = min(r + pad, w)
                b_padded = min(b + pad, h)
                face_roi = frame_resized[t_padded:b_padded, l_padded:r_padded]

                if face_roi.size == 0:
                    continue

                student_id_display = f"" # Default display is track ID

                if face_recognition_enabled and face_roi.size != 0:
                    try:
                        # Face recognition using DeepFace.find
                        dfs = DeepFace.find(face_roi, db_path=student_images_dir, enforce_detection=False, silent=True)
                        if dfs: # If faces are found
                            df = dfs[0] # Consider the first result for simplicity
                            if not df.empty:
                                identity = df['identity'].values[0]
                                # Extract student ID from the image filename
                                recognized_student_name = os.path.basename(identity).split('.')[0] # Extract name from matched image filename
                                student_id_display = f"{recognized_student_name}" # Display student name
                            else:
                                print(f"No face match found for track ID {track_id}")
                                student_id_display = "" # or "Unrecognized" if no match
                        else:
                             print(f"No face found in face_roi for track ID {track_id} during recognition attempt.")
                             student_id_display = "" # or "Unrecognized" if no face found for recognition

                    except Exception as e:
                        print(f"Face recognition error: {e}")
                        student_id_display = f"Unrecognized ID {track_id} (Recognition Error)" # Indicate recognition error, but still mark as unrecognized

                # If the face is frontal, perform emotion detection;
                # otherwise, simply set the emotion label to "not frontal".
                if is_face_frontal(face_roi):
                    try:
                        # DeepFace.analyze returns a list when multiple faces are detected.
                        analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                        # If multiple results are returned, pick the first.
                        if isinstance(analysis, list):
                            analysis = analysis[0]
                        dominant_emotion = analysis['dominant_emotion']
                        emotion_confidence = analysis['emotion'][dominant_emotion]
                        # If the confidence is low, set the emotion as unknown.
                        if emotion_confidence < 0.7:
                            dominant_emotion = 'unknown'
                    except Exception as e:
                        dominant_emotion = 'unknown'
                else:
                    dominant_emotion = 'not frontal'

                # Update emotion duration (in seconds) for the current track and emotion label.
                emotion_dict[track_id][dominant_emotion] += 1 / fps

                # Write the log data to CSV
                writer.writerow({
                    'frame': frame_count,
                    'student_id': student_id_display,
                    'emotion': dominant_emotion,
                    'emotion_duration': emotion_dict[track_id][dominant_emotion]
                })

                # Display the information on the frame
                cv2.putText(frame_resized, f"{student_id_display} | {dominant_emotion}",
                            (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame_resized, (l, t), (r, b), (255, 0, 0), 2)

            # Display the frame
            cv2.imshow('Real-Time Emotion Tracking', frame_resized)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

# Summary Report
print(f"Emotion data logged in real-time to {csv_filename}")
for student_id, emotions in emotion_dict.items():
    print(f"\nStudent {student_id} Emotion Summary:")
    for emotion, duration in emotions.items():
        print(f" - {emotion}: {duration:.2f} seconds")
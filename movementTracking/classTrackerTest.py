import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import csv
import os
from deepface import DeepFace  # Import DeepFace
import torch # Import torch
from collections import defaultdict # Import defaultdict

# -----------------------------
# Configuration
# -----------------------------
IOU_THRESHOLD = 0.5
NUM_STUDENTS = 5  # Initial number of students (now dynamically tracked)
EMOTION_CONFIDENCE_THRESHOLD = 0.7 # Threshold for emotion confidence

# -----------------------------
# Set DeepFace model home directory
# -----------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
print("base_dir: ", base_dir)
model_dir = os.path.join(base_dir, 'model')
os.environ['DEEPFACE_HOME'] = model_dir
print(f"DeepFace models will be stored in: {model_dir}")

# -----------------------------
# Explicitly create DeepFace weights directory
# -----------------------------
weights_dir = os.path.join(model_dir, '.deepface', 'weights')
os.makedirs(weights_dir, exist_ok=True) # Create directory, no error if exists
print(f"Ensuring DeepFace weights directory exists: {weights_dir}")

# -----------------------------
# Hardware Acceleration Setup
# -----------------------------
# Check for MPS availability (Apple Silicon Macs)
if torch.backends.mps.is_available():
    device = "mps"
    print("Using MPS for hardware acceleration.")
# Check for CUDA availability (NVIDIA GPUs)
elif torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA for hardware acceleration.")
# Fallback to CPU if neither MPS nor CUDA is available
else:
    device = "cpu"
    print("Using CPU. Hardware acceleration not available.")

# -----------------------------
# Initialize Models and Utilities
# -----------------------------
model_path = os.path.join(base_dir, 'model', 'yolov8n.pt')
video_path = os.path.join(base_dir, 'videos', 'school_kid_class1.mp4')
csv_filename = os.path.join(base_dir, 'logs', 'real_time_student_emotion_log.csv') # Changed log filename
image_folder_path = os.path.join(base_dir, 'images') # Define image folder path

# Load YOLO model and tracker
yolo_model = YOLO(model_path, device=device)
tracker = DeepSort(max_age=30, n_init=5, device=device)

cap = cv2.VideoCapture(video_path)
movement_dict = {}
student_name_map = {} # Dictionary to map track_id to student name
emotion_dict = defaultdict(lambda: defaultdict(float)) # Dictionary to store emotion data
fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_count = 0
identification_done = False # Flag to control face identification

# Placeholder function to simulate loading student data from database
def load_student_data(image_folder_path):
    student_data = {}
    for i in range(1, NUM_STUDENTS + 1):
        student_id = i
        student_name = f"Student {i}"
        image_path = os.path.join(image_folder_path, f'student_{i}.jpg') # Assuming images are in 'images' folder
        student_data[student_id] = {'student_name': student_name, 'image_path': image_path}
    return student_data

student_data = load_student_data(image_folder_path) # Load student data at the beginning

def recognize_student_faces(frame, detections, student_data, base_dir): # Added base_dir
    face_recognition_results = {}
    for i, detection in enumerate(detections): # detections is a list of (bbox, conf, None)
        bbox, conf, _ = detection
        x, y, w, h = bbox
        face_roi = frame[y:y+h, x:x+w] # Extract face ROI from the detection bbox

        try:
            recognition = DeepFace.find(img_path = face_roi,
                                        db_path = os.path.join(base_dir, 'images'), # Path to student images directory
                                        enforce_detection=False, # If face is already provided as ROI, no need to detect again
                                        silent=True) # Suppress output

            if recognition and isinstance(recognition, list) and len(recognition) > 0: # Check if recognition is a list and not empty
                recognition_result = recognition[0] # Take the first result from the list
                if isinstance(recognition_result, dict) and 'confidence' in recognition_result and 'identity' in recognition_result: # Check if result is a dict and has 'confidence' and 'identity' keys
                    confidence = recognition_result['confidence'][0] # Get the confidence score
                    identity = recognition_result['identity'][0] # Get the first identity from the list
                    student_filename = os.path.basename(identity) # Extract filename
                    student_id_str = student_filename.split('_')[1].split('.')[0] # Extract student ID from filename (e.g., student_1.jpg -> 1)
                    try:
                        student_id = int(student_id_str)
                        if student_id in student_data:
                            face_recognition_results[i] = student_id # Map detection index to student_id
                            print(f"Detection {i}: Face recognized as Student {student_id} with confidence {confidence:.2f}") # Print student ID and confidence
                        else:
                            print(f"Detection {i}: Student ID {student_id} not found in student data.")
                            face_recognition_results[i] = None
                    except ValueError:
                        print(f"Detection {i}: Could not parse student ID from filename: {student_filename}")
                        face_recognition_results[i] = None
                else:
                    print(f"Detection {i}: Unexpected recognition result format: {recognition_result}") # Print unexpected result
                    face_recognition_results[i] = None
            else:
                print(f"Detection {i}: No face match found or empty recognition list: {recognition}") # Print no match message and recognition result
                face_recognition_results[i] = None # No match found
        except Exception as e:
            print(f"Detection {i}: Face recognition error: {e}")
            face_recognition_results[i] = None # Error during recognition

    return face_recognition_results

# Face orientation detection (ensures frontal faces) - from euan.py
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

def analyze_student_emotion(face_roi): # New function for emotion analysis
    dominant_emotion = 'unknown'
    try:
        analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, silent=True) # Added silent=True
        if isinstance(analysis, list):
            analysis = analysis[0]
        dominant_emotion = analysis['dominant_emotion']
        emotion_confidence = analysis['emotion'][dominant_emotion]
        if emotion_confidence < EMOTION_CONFIDENCE_THRESHOLD: # Use threshold from config
            dominant_emotion = 'unknown'
    except Exception as e:
        print(f"Emotion analysis error: {e}")
        dominant_emotion = 'unknown'
    return dominant_emotion

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

def update_tracker_and_recognize_faces(frame_resized, frame, frame_count, identification_done, student_data, movement_dict, student_name_map, tracker, base_dir): # Added base_dir
    # YOLO detection
    results = yolo_model(frame_resized)[0]
    detections = []
    for result in results.boxes.data:
        x1, y1, x2, y2, conf, cls = result
        if int(cls) == 0 and conf > 0.5:
            bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1), ]
            detections.append((bbox, conf, None))

    # Update tracker with detections
    tracks = tracker.update_tracks(detections, frame=frame_resized)

    # Face recognition (performed once at the beginning)
    if not identification_done:
        face_recognition_map = recognize_student_faces(frame_resized, detections, student_data, base_dir) # Pass base_dir
        for detection_index, student_id in face_recognition_map.items():
            if student_id is not None and detection_index < len(tracks): # Check if detection_index is within track bounds
                track = tracks[detection_index]
                student_name = student_data[student_id]['student_name']
                student_name_map[track.track_id] = student_name # Map track_id to student name
                if track.track_id in movement_dict: # Update student_name in movement_dict if track_id is already there
                    movement_dict[track.track_id]['student_name'] = student_name
        return True # Identification is now done (for this initial phase)

    return identification_done

def calculate_movement_emotion_and_log(frame_count, frame_resized, tracks, movement_dict, student_name_map, writer, emotion_dict, fps): # Added emotion_dict, fps, frame_resized
    h, w, _ = frame_resized.shape # Get frame dimensions for boundary checks
    for track in tracks:
        if not track.is_confirmed() or track.hits < 3:
            continue

        track_id = track.track_id
        l, t, r, b = track.to_ltrb()

        # Boundary check and ensure integers (from euan.py)
        l = max(0, int(l))
        t = max(0, int(t))
        r = min(w, int(r))
        b = min(h, int(b))

        # Optional: expand the bounding box slightly to capture more of the face. (from euan.py)
        pad = 10  # pixels
        l_padded = max(l - pad, 0)
        t_padded = max(t - pad, 0)
        r_padded = min(r + pad, w)
        b_padded = min(b + pad, h)
        face_roi = frame_resized[t_padded:b_padded, l_padded:r_padded]

        if face_roi.size == 0: # Skip if face ROI is empty
            continue

        dominant_emotion = 'not frontal' # Default emotion
        if is_face_frontal(face_roi): # Check for frontal face
            dominant_emotion = analyze_student_emotion(face_roi) # Analyze emotion if frontal

        emotion_dict[track_id][dominant_emotion] += 1 / fps # Update emotion duration

        matched_id = None # Movement logic - same as before
        for existing_id, data in movement_dict.items():
            if calculate_iou( [data['bbox'][0], data['bbox'][1], data['bbox'][0] + data['bbox'][2], data['bbox'][1] + data['bbox'][3]], [l, t, r, b]) > IOU_THRESHOLD: # Convert bbox format for IoU
                matched_id = existing_id
                break

        if matched_id is None:
            matched_id = track_id
            movement_dict[matched_id] = {
                'prev_position': np.array([(l + r) / 2, (t + b) / 2]),
                'cumulative_movement': 0,
                'bbox': [l, t, r-l, b-t], # Store bbox as [x, y, w, h]
                'student_name': student_name_map.get(track_id, None), # Get student name from map
                'dominant_emotion': dominant_emotion # Store dominant emotion
            }
            movement_delta = 0
        else:
            prev_position = movement_dict[matched_id]['prev_position']
            current_position = np.array([(l + r) / 2, (t + b) / 2])
            movement_delta = np.linalg.norm(current_position - prev_position)
            movement_dict[matched_id]['cumulative_movement'] += movement_delta
            movement_dict[matched_id]['prev_position'] = current_position
            movement_dict[matched_id]['bbox'] = [l, t, r-l, b-t] # Update bbox
            movement_dict[matched_id]['dominant_emotion'] = dominant_emotion # Update dominant emotion

        writer.writerow({ # Log updated to include emotion and duration
            'frame': frame_count,
            'student_id': matched_id,
            'student_name': student_name_map.get(matched_id, ""), # Log student name
            'movement_delta': movement_delta,
            'cumulative_movement': movement_dict[matched_id]['cumulative_movement'],
            'emotion': dominant_emotion,
            'emotion_duration': emotion_dict[matched_id][dominant_emotion]
        })
    return movement_dict, emotion_dict

def visualize_frame(frame_resized, tracks, movement_dict, student_name_map):
    for track in tracks:
        if not track.is_confirmed() or track.hits < 3:
            continue

        track_id = track.track_id
        l, t, r, b = track.to_ltrb()

        student_name = movement_dict[track_id]['student_name'] if track_id in movement_dict and movement_dict[track_id]['student_name'] else ""
        dominant_emotion = movement_dict[track_id]['dominant_emotion'] if track_id in movement_dict and 'dominant_emotion' in movement_dict[track_id] else "unknown" # Get emotion

        label_text = f"ID:{track_id} {student_name} {dominant_emotion} {movement_dict[track_id]['cumulative_movement']:.2f}" # Combined label
        cv2.putText(frame_resized, label_text,
                    (int(l), int(t) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(frame_resized, (int(l), int(t)), (int(r), int(b)), (255, 0, 0), 2)
    return frame_resized


# Ensure the logs directory exists
os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

# Initialize CSV file for real-time logging
with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=['frame', 'student_id', 'student_name', 'movement_delta', 'cumulative_movement', 'emotion', 'emotion_duration']) # Updated fieldnames
    writer.writeheader()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_resized = cv2.resize(frame, (640, 480))

            identification_done = update_tracker_and_recognize_faces(frame_resized, frame, frame_count, identification_done, student_data, movement_dict, student_name_map, tracker, base_dir) # Pass base_dir
            movement_dict, emotion_dict = calculate_movement_emotion_and_log(frame_count, frame_resized, tracks, movement_dict, student_name_map, writer, emotion_dict, fps) # Pass frame_resized, emotion_dict, fps
            frame_resized_with_boxes = visualize_frame(frame_resized, tracks, movement_dict, student_name_map)


            cv2.imshow('Real-Time Movement Tracking with Emotion', frame_resized_with_boxes) # Updated window title
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

# Emotion Summary Report at the end (from euan.py - adapted for student names)
print(f"Motion and emotion data logged in real-time to {csv_filename}")
for student_id, emotions in emotion_dict.items():
    student_name = student_name_map.get(student_id, f"Student {student_id}") # Get student name or default
    print(f"\nStudent ID: {student_id}, Name: {student_name} - Emotion Summary:")
    for emotion, duration in emotions.items():
        print(f" - {emotion}: {duration:.2f} seconds")

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from deepface import DeepFace
import csv
import os
from collections import defaultdict
import pickle

# -----------------------------
# Helper Functions for Embeddings
# -----------------------------
def build_embedding_database(student_images_dir, model_name='Facenet512', detector_backend='retinaface'):
    """
    Precompute embeddings for each student image in the directory.
    Returns a dict mapping {student_name: embedding_vector}.
    """
    embedding_database = {}

    if not os.path.exists(student_images_dir):
        os.makedirs(student_images_dir)
        print(f"Please put student images (e.g., name.jpg) in '{student_images_dir}'")
        return embedding_database

    for filename in os.listdir(student_images_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            student_name = os.path.splitext(filename)[0]  # "alice.jpg" -> "alice"
            img_path = os.path.join(student_images_dir, filename)
            try:
                # DeepFace.represent returns a list of dicts (one per face).
                # We'll assume there's only one face in each student image.
                embedding_objs = DeepFace.represent(
                    img_path=img_path,
                    model_name=model_name,
                    enforce_detection=False,  # set True if images are guaranteed to have a clear face
                    detector_backend=detector_backend
                )
                if embedding_objs:
                    embedding_database[student_name] = embedding_objs[0]['embedding']
                    print(f"[Embedding] Computed embedding for {student_name}.")
            except Exception as e:
                print(f"[Warning] Could not compute embedding for {filename}: {e}")

    return embedding_database


def cosine_similarity(vec1, vec2):
    """
    Returns the cosine similarity between two vectors.
    Similarity ranges from -1.0 to 1.0, where 1.0 is perfectly similar.
    """
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)


def recognize_face(face_roi, embedding_database, model_name='Facenet512', threshold=0.55):
    """
    Given a cropped face (face_roi) and a precomputed embedding database,
    compute the face's embedding using DeepFace, then compare it (cosine distance)
    with each student's embedding in `embedding_database`.

    'threshold' is the cosine distance below which we consider it a match.
    (cosine distance = 1 - cosine similarity).

    Returns the recognized student name or 'Unknown'.
    """
    try:
        face_embedding_objs = DeepFace.represent(
            face_roi,
            model_name=model_name,
            enforce_detection=False
        )
        if not face_embedding_objs:
            return "No Face Found"

        face_embedding = face_embedding_objs[0]['embedding']

        min_distance = float('inf')
        best_match = None

        for student_name, db_embedding in embedding_database.items():
            # Compute cosine distance = 1 - cosine_similarity
            sim = cosine_similarity(face_embedding, db_embedding)
            cos_distance = 1 - sim

            if cos_distance < min_distance:
                min_distance = cos_distance
                best_match = student_name

        # Check if the minimum distance is below threshold
        if min_distance < threshold:
            return best_match
        else:
            return "Unknown"

    except Exception as e:
        print(f"[Error] Face recognition error: {e}")
        return "Error"


# -----------------------------
# Main Script
# -----------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))

# Paths
model_path = os.path.join(base_dir, 'model', 'yolov8n.pt')
video_path = os.path.join(base_dir, 'videos', 'zoom.mp4')
csv_filename = os.path.join(base_dir, 'logs', 'real_time_emotion_log.csv')
student_images_dir = os.path.join(base_dir, 'images')  # Directory containing student images
output_videos_dir = os.path.join(base_dir, 'outputVideos')

# Ensure directories exist
os.makedirs(os.path.join(base_dir, 'logs'), exist_ok=True)
os.makedirs(output_videos_dir, exist_ok=True)
if not os.path.exists(student_images_dir):
    os.makedirs(student_images_dir)
    print(f"Please put student images (e.g., name.jpg) in '{student_images_dir}'")

# Build the embedding database from images in student_images_dir
# (Adjust model_name if you prefer a different model, e.g. 'ArcFace', 'VGG-Face', etc.)
embedding_database = build_embedding_database(student_images_dir, model_name='Facenet512')
face_recognition_enabled = len(embedding_database) > 0
print(f"[Info] Face recognition enabled: {face_recognition_enabled} (Database size: {len(embedding_database)})")

# Load YOLO model and DeepSort tracker
yolo_model = YOLO(model_path)
tracker = DeepSort(max_age=30, n_init=5)

cap = cv2.VideoCapture(video_path)

NUM_STUDENTS = 26
movement_dict = {}  # To store unique track IDs
emotion_dict = defaultdict(lambda: defaultdict(float))

fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_count = 0
record_duration_sec = 60  # e.g., record for 60 seconds

# Initialize VideoWriter for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = os.path.join(output_videos_dir, 'output_video_fast.mp4')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (640, 480))

# Face orientation detection (ensures frontal faces) - optional
def is_face_frontal(face_roi):
    """
    Using adjusted parameters for a more lenient detection.
    scaleFactor=1.5 and minNeighbors=2 means the cascade is less strict,
    which can detect more faces even if they are not perfectly frontal.
    """
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=2)
    return len(faces) > 0

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
            elapsed_time_sec = frame_count / fps
            if elapsed_time_sec > record_duration_sec:
                print(f"Reached {record_duration_sec} seconds of recording. Stopping.")
                break

            # Resize frame for processing; adjust size if needed.
            frame_resized = cv2.resize(frame, (640, 480))
            h, w, _ = frame_resized.shape

            # 1. YOLO detection
            results = yolo_model(frame_resized)[0]

            # 2. Convert YOLO detections to DeepSort format
            detections = []
            for result in results.boxes.data:
                x1, y1, x2, y2, conf, cls = result
                # Only select persons (class 0) with confidence > 0.5
                if int(cls) == 0 and conf > 0.5:
                    bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]  # x, y, w, h
                    detections.append((bbox, float(conf), None))

            # 3. Update tracks
            tracks = tracker.update_tracks(detections, frame=frame_resized)

            # 4. Process each track
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id

                # Keep track of unique student IDs, up to NUM_STUDENTS
                if track_id not in movement_dict:
                    if len(movement_dict) < NUM_STUDENTS:
                        movement_dict[track_id] = True
                    else:
                        # Skip new tracks if we've already reached the limit
                        continue

                # Convert track bounding box to integer l, t, r, b
                l, t, r, b = track.to_ltrb()
                l = max(0, int(l))
                t = max(0, int(t))
                r = min(w, int(r))
                b = min(h, int(b))

                # Optional: expand the bounding box slightly to capture more of the face
                pad = 10
                l_padded = max(l - pad, 0)
                t_padded = max(t - pad, 0)
                r_padded = min(r + pad, w)
                b_padded = min(b + pad, h)
                face_roi = frame_resized[t_padded:b_padded, l_padded:r_padded]

                if face_roi.size == 0:
                    continue

                # Default display is just the track_id
                student_id_display = f"{track_id}"

                # If face recognition is enabled, attempt to recognize
                if face_recognition_enabled:
                    recognized_student_name = recognize_face(
                        face_roi, 
                        embedding_database, 
                        model_name='Facenet512', 
                        threshold=0.3  # Cosine distance threshold
                    )
                    if recognized_student_name not in ["Unknown", "No Face Found", "Error"]:
                        student_id_display = recognized_student_name
                    else:
                        # Optionally keep track ID or say "Unknown"
                        student_id_display = "Unknown"

                # Face orientation check for emotion analysis
                if is_face_frontal(face_roi):
                    try:
                        # DeepFace.analyze might return a list if multiple faces are found.
                        analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
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

                # Update emotion duration (in seconds) for the current track/emotion
                emotion_dict[track_id][dominant_emotion] += 1 / fps

                # Write log data to CSV
                writer.writerow({
                    'frame': frame_count,
                    'student_id': student_id_display,
                    'emotion': dominant_emotion,
                    'emotion_duration': emotion_dict[track_id][dominant_emotion]
                })

                # Display info on the frame
                cv2.putText(frame_resized, f"{student_id_display} | {dominant_emotion}",
                            (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame_resized, (l, t), (r, b), (255, 0, 0), 2)

            # Write processed frame to the output video
            video_writer.write(frame_resized)

            # Optionally display the frame in a window
            cv2.imshow('Real-Time Emotion Tracking', frame_resized)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

# Summary Report
print(f"\n[Info] Emotion data logged to {csv_filename}")
for student_id, emotions in emotion_dict.items():
    print(f"\nStudent/Track {student_id} Emotion Summary:")
    for emotion, duration in emotions.items():
        print(f" - {emotion}: {duration:.2f} seconds")

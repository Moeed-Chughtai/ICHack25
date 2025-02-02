import cv2
import numpy as np
import os
import csv
from collections import defaultdict

# YOLO and DeepSort
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# DeepFace
from deepface import DeepFace

# Concurrency for face recognition/emotion analysis
from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------------
# Initialize Models and Utilities
# -----------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_dir, 'model', 'yolov8n.pt')
video_path = os.path.join(base_dir, 'videos', 'zoom.mp4')
csv_filename = os.path.join(base_dir, 'logs', 'real_time_emotion_log.csv')
student_images_dir = os.path.join(base_dir, 'images')  # Directory containing student images
output_videos_dir = os.path.join(base_dir, 'outputVideos')
os.makedirs(output_videos_dir, exist_ok=True)

NUM_STUDENTS = 26
record_duration_sec = 60

# Initialize YOLO and DeepSort
yolo_model = YOLO(model_path)
tracker = DeepSort(max_age=30, n_init=5)

# Initialize dictionary to store (track_id -> recognized info)
# recognized_students_info[track_id] = { 'name': str or None, 'distance': float }
recognized_students_info = {}

# Initialize dictionary to store (track_id -> { emotion_label: total_seconds })
emotion_dict = defaultdict(lambda: defaultdict(float))

# Movement dict to limit unique track IDs to NUM_STUDENTS
movement_dict = {}

# Collect student images (for face recognition) into a dict
student_name_dict = {}
if not os.path.exists(student_images_dir):
    os.makedirs(student_images_dir)
    print(f"Please put student images (name.jpg) in '{student_images_dir}'")
else:
    for filename in os.listdir(student_images_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            student_name = os.path.splitext(filename)[0]
            student_name_dict[student_name] = filename

face_recognition_enabled = bool(student_name_dict)
print(f"Face recognition enabled: {face_recognition_enabled}")

# Video capture
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_count = 0

# Setup VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = os.path.join(output_videos_dir, 'output_video.mp4')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (640, 480))

# Make sure the logs directory exists
os.makedirs(os.path.join(base_dir, 'logs'), exist_ok=True)

# ------------------------------------------------------------------------------
# Helper function to do face recognition + emotion detection (heavy-lifting)
# ------------------------------------------------------------------------------
def analyze_face(
    face_roi_bgr,                # The face crop (OpenCV BGR image)
    track_id,                    # Track ID
    recognized_students_info,    # Current dict of recognized info
    student_images_dir,          # Directory of reference images
    face_recognition_enabled,    # Bool
    distance_threshold=0.55,     # Distance threshold for recognition
    emotion_confidence_min=0.7   # Minimum confidence to accept dominant emotion
):
    """
    This function is meant to run in a separate process (via ProcessPoolExecutor).
    It returns (track_id, recognized_name, recognized_distance, dominant_emotion).
    """
    # Default outputs
    recognized_name = None
    best_dist = float('inf')
    dominant_emotion = 'neutral'

    # Attempt face recognition
    recognized_info = recognized_students_info.get(track_id, {'name': None, 'distance': float('inf')})
    best_distance_so_far = recognized_info['distance']

    if face_recognition_enabled and face_roi_bgr.size != 0:
        # If we don't have a confident match yet, or we want to re-check
        if recognized_info['name'] is None or best_distance_so_far >= 0.15:
            try:
                results_list = DeepFace.find(
                    face_roi_bgr,
                    db_path=student_images_dir,
                    enforce_detection=False,
                    silent=True,
                    threshold=distance_threshold,
                    model_name="Facenet512",
                    detector_backend="retinaface"
                )
                if results_list:  # If DB search returned results
                    df = results_list[0]
                    if not df.empty:
                        df = df.sort_values(by='distance', ascending=True)
                        best_dist = df.iloc[0]['distance']
                        identity = df.iloc[0]['identity']
                        if best_dist < distance_threshold and best_dist < best_distance_so_far:
                            recognized_name = os.path.basename(identity).split('.')[0]
                # Otherwise, recognized_name remains None
            except Exception as e:
                # If face recognition fails, just leave recognized_name as None
                pass
        else:
            # Already have a recognized name with a decent distance
            recognized_name = recognized_info['name']
            best_dist = recognized_info['distance']
    else:
        # Face recognition not enabled or empty ROI
        recognized_name = recognized_info['name']  # Might be None
        best_dist = recognized_info['distance']

    # Next, do emotion detection if face is present
    # (We rely on the main thread to do 'is_face_frontal' check to reduce wasted calls.)
    try:
        analysis = DeepFace.analyze(face_roi_bgr, actions=['emotion'], enforce_detection=False)
        if isinstance(analysis, list):
            analysis = analysis[0]
        dom_emotion = analysis['dominant_emotion']
        conf = analysis['emotion'][dom_emotion]
        if conf >= emotion_confidence_min:
            dominant_emotion = dom_emotion
        else:
            dominant_emotion = 'unknown'
    except:
        dominant_emotion = 'unknown'

    return track_id, recognized_name, best_dist, dominant_emotion

# ------------------------------------------------------------------------------
# Face Orientation Check
# ------------------------------------------------------------------------------
def is_face_frontal(face_roi):
    """
    Returns True if the face ROI is frontal enough.
    """
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=2)
    return len(faces) > 0

# ------------------------------------------------------------------------------
# Main Loop
# ------------------------------------------------------------------------------
with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=['frame', 'student_id', 'emotion', 'emotion_duration'])
    writer.writeheader()

    # We'll use a ProcessPoolExecutor to parallelize DeepFace calls:
    # Adjust the number of workers to match your GPU/CPU resources.
    with ProcessPoolExecutor(max_workers=8) as executor:
        # Keep track of outstanding futures
        future_to_track_id = {}

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

                # Resize frame
                frame_resized = cv2.resize(frame, (640, 480))
                h, w, _ = frame_resized.shape

                # YOLO detection
                results = yolo_model(frame_resized)[0]
                detections = []
                for result in results.boxes.data:
                    x1, y1, x2, y2, conf, cls = result
                    if int(cls) == 0 and conf > 0.5:
                        bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                        detections.append((bbox, float(conf), None))

                # Update DeepSort tracks
                tracks = tracker.update_tracks(detections, frame=frame_resized)

                # Collect any finished futures before submitting new ones
                done_futures = []
                for fut in list(future_to_track_id.keys()):
                    if fut.done():
                        done_futures.append(fut)

                # Process completed futures
                for fut in done_futures:
                    track_id = future_to_track_id.pop(fut)
                    try:
                        tid, recognized_name, best_dist, emotion = fut.result()
                    except Exception as e:
                        # If an error happened in the worker
                        print(f"[Worker Error] Track ID {track_id}: {e}")
                        continue

                    # Update recognized info
                    old_info = recognized_students_info.get(tid, {'name': None, 'distance': float('inf')})
                    # If the new distance is better, or we didn't have a name yet, update
                    if recognized_name is not None and best_dist < old_info['distance']:
                        recognized_students_info[tid] = {
                            'name': recognized_name,
                            'distance': best_dist
                        }
                    # Update emotion data
                    emotion_dict[tid][emotion] += 1 / fps

                    # Write a row to CSV
                    # Display name
                    disp_name = recognized_students_info[tid]['name'] if recognized_students_info[tid]['name'] else ""
                    writer.writerow({
                        'frame': frame_count,
                        'student_id': disp_name,
                        'emotion': emotion,
                        'emotion_duration': emotion_dict[tid][emotion]
                    })

                # Now handle the current tracks
                for track in tracks:
                    if not track.is_confirmed():
                        continue

                    track_id = track.track_id
                    # Enforce limit of NUM_STUDENTS
                    if track_id not in movement_dict:
                        if len(movement_dict) < NUM_STUDENTS:
                            movement_dict[track_id] = True
                        else:
                            # Skip new IDs if we have reached limit
                            continue

                    l, t, r, b = track.to_ltrb()
                    l = max(0, int(l))
                    t = max(0, int(t))
                    r = min(w, int(r))
                    b = min(h, int(b))

                    pad = 10
                    l_padded = max(l - pad, 0)
                    t_padded = max(t - pad, 0)
                    r_padded = min(r + pad, w)
                    b_padded = min(b + pad, h)
                    face_roi = frame_resized[t_padded:b_padded, l_padded:r_padded]

                    if face_roi.size == 0:
                        continue

                    # We'll do a quick frontal check here to avoid wasted worker calls
                    if is_face_frontal(face_roi):
                        # Submit analysis to the worker if we don't already have a pending future for this track
                        # Or if you want to analyze every frame, remove the "if track_id not in future_to_track_id" check.
                        # That said, analyzing every single frame might be expensive. You can also skip frames or check
                        # if a prior future for the same track_id is still running. (Up to your logic.)
                        if track_id not in future_to_track_id.values():
                            # Make a copy of recognized_students_info for the worker to read
                            # (Because Python's ProcessPool doesn't share memory by default.)
                            # We'll pass minimal data needed:
                            recognized_info_copy = recognized_students_info.copy()

                            # Submit
                            future = executor.submit(
                                analyze_face,
                                face_roi,
                                track_id,
                                recognized_info_copy,
                                student_images_dir,
                                face_recognition_enabled
                            )
                            future_to_track_id[future] = track_id

                        # For quick UI, we *could* show the last known data (not the new future's result).
                        current_name = recognized_students_info.get(track_id, {}).get('name', "")
                        # We don't know the new emotion yet, so let's just show "Analyzing..."
                        cv2.putText(
                            frame_resized,
                            f"{current_name} | Analyzing...",
                            (l, t - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )
                    else:
                        # If not frontal, we won't do face recognition this frame
                        # We can store a default "neutral" or "not_frontal" if we like
                        cv2.putText(
                            frame_resized,
                            f"Not Frontal",
                            (l, t - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            2
                        )

                    # Draw bounding box
                    cv2.rectangle(frame_resized, (l, t), (r, b), (255, 0, 0), 2)

                # Write frame to output video
                video_writer.write(frame_resized)

                # Display the frame
                cv2.imshow('Real-Time Emotion Tracking (Distributed)', frame_resized)
                if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            video_writer.release()
            cv2.destroyAllWindows()

# Summary (at end)
print(f"Emotion data logged in real-time to {csv_filename}")
for student_id, emotions in emotion_dict.items():
    # If you had a recognized name, it might be recognized_students_info[student_id]['name']
    recognized_name = recognized_students_info.get(student_id, {}).get('name', None)
    disp_id = recognized_name if recognized_name else student_id
    print(f"\nSummary for {disp_id}:")
    for emotion, duration in emotions.items():
        print(f" - {emotion}: {duration:.2f} seconds")

import pandas as pd
import matplotlib.pyplot as plt

# Load the motion log CSV file
csv_filename = 'real_time_motion_log.csv'
data = pd.read_csv(csv_filename)

# Convert cumulative movement to per-frame movement
students = data['student_id'].unique()

# Create a dictionary to hold per-frame movements
per_frame_movement = {student: [] for student in students}

# Process data to calculate per-frame movement
data.sort_values(by=['student_id', 'frame'], inplace=True)

for student in students:
    student_data = data[data['student_id'] == student]
    prev_cumulative = 0
    for index, row in student_data.iterrows():
        # Calculate per-frame movement
        per_frame = row['cumulative_movement'] - prev_cumulative
        per_frame_movement[student].append((row['frame'], per_frame))
        prev_cumulative = row['cumulative_movement']

# Plotting the per-frame movement
plt.figure(figsize=(12, 8))
for student, movements in per_frame_movement.items():
    frames = [frame for frame, movement in movements]
    movement_values = [movement for frame, movement in movements]
    plt.plot(frames, movement_values, label=f'Student {student}')

plt.xlabel('Frame')
plt.ylabel('Movement')
plt.title('Per-Frame Movement of Each Student Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

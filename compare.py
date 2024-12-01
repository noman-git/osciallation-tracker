import pandas as pd
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import argparse

# Load the apparatus data
data = pd.read_csv("output_pos.csv", header=None)
data.columns = ["Time", "Value"]

# Convert columns to numeric, handling non-numeric gracefully
data["Time"] = pd.to_numeric(data["Time"], errors="coerce")
data["Value"] = pd.to_numeric(data["Value"], errors="coerce")

# Drop any rows with NaN values (caused by non-numeric entries)
data.dropna(inplace=True)

# Get the total duration of the data in seconds
total_duration = data["Time"].max()

# Number of points to keep (30 points for 30 FPS)
num_points_to_keep = int(total_duration * 30)

# Downsample the apparatus data to 30 FPS
downsampled_data = data.sample(n=num_points_to_keep,
                               random_state=1)  # Randomly sample points
downsampled_data.sort_values(by="Time",
                             inplace=True)  # Sort by time for plotting

# Set up argument parsing for YOLO
parser = argparse.ArgumentParser(
    description="YOLO object tracking on a video.")
parser.add_argument("video_path",
                    type=str,
                    help="Path to the video file to be processed")
args = parser.parse_args()

# Load the fine-tuned YOLO model
model = YOLO("model/custom_yolo.pt")

# Initialize video capture
cap = cv2.VideoCapture(args.video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 0  # Initialize frame time counter

# Lists to store tracking data
y_positions = []
frame_times = []

# Get the class ID for 'crash_test_sign'
crash_test_sign_id = next(key for key, value in model.names.items()
                          if value == "crash_test_sign")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection on the frame
    results = model(frame, verbose=False)
    detections = results[0].boxes  # Access detected bounding boxes

    # Filter for 'crash_test_sign' detections using the class ID
    crash_test_signs = [
        box for box in detections if box.cls == crash_test_sign_id
    ]

    if crash_test_signs:
        # Define tracking criterion (e.g., closest to the center)
        frame_center_x, frame_center_y = frame.shape[1] / 2, frame.shape[0] / 2

        # Find the object closest to the center
        closest_sign = min(
            crash_test_signs,
            key=lambda box: ((box.xywh[0][0] - frame_center_x)**2 +
                             (box.xywh[0][1] - frame_center_y)**2).sum(),
        )

        # Extract coordinates
        x1, y1, x2, y2 = map(int, closest_sign.xyxy[0])

        # Track the vertical center of the bounding box
        center_y = (y1 + y2) // 2
        y_positions.append(center_y)
        frame_times.append(frame_time)

    # Update frame time based on FPS
    frame_time += 1 / fps

# Release video resources
cap.release()

# Downsample YOLO tracking data to match 30 FPS for comparison
y_positions = y_positions[:num_points_to_keep]  # Ensure we have enough points
frame_times = frame_times[:num_points_to_keep]

# Shift YOLO tracking data by 2 seconds to the right
# This has no bearing on the experiment, it is for the
# of comparison only
shifted_frame_times = [t + 2.06 for t in frame_times]

frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Invert y_positions by subtracting each position from the frame height
inverted_y_positions = [frame_height - y for y in y_positions]

# Normalize the apparatus data by centering it around 0
apparatus_mean = downsampled_data["Value"].mean()
normalized_apparatus_data = downsampled_data["Value"] - apparatus_mean

# Normalize the YOLO tracking data by centering it around 0
yolo_mean = sum(inverted_y_positions) / len(inverted_y_positions)
normalized_yolo_positions = [y - yolo_mean for y in inverted_y_positions]

# Shift the normalized YOLO positions up by 88 units
# Again this is just for comparison purposes. What matters
# is the relative amplitude
shifted_normalized_yolo_positions = [y + 37 for y in normalized_yolo_positions]

# Create the comparative plot
plt.figure(figsize=(10, 5))

# Plot the normalized apparatus data
plt.plot(
    downsampled_data["Time"],
    normalized_apparatus_data,
    color="blue",
    linewidth=2,
    label="Apparatus Data",
)

# Plot the shifted and normalized YOLO tracking data
plt.plot(
    shifted_frame_times,
    shifted_normalized_yolo_positions,
    label="YOLO Tracking Data",
    color="orange",
    linewidth=2,
)

# Label the axes
plt.xlabel("Time (seconds)")
plt.ylabel("Normalized Value")

# Add a title
plt.title("Comparative Plot of Normalized Apparatus and YOLO Tracking Data")

# Add legend
plt.legend()

# Display a grid for readability
plt.grid(True)

# Show the plot
plt.show()

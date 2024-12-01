import cv2
import os
from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(
    description="Crop video based on crash test sign's motion."
)
parser.add_argument("video_path", type=str, help="Path to the video file to be cropped")
args = parser.parse_args()

model = YOLO("model/custom_yolo.pt")

output_dir = "cropped_vid"
os.makedirs(output_dir, exist_ok=True)

# Output filename with '_cropped' suffix
video_name = os.path.basename(args.video_path).split(".")[0] + "_cropped.mp4"
output_path = os.path.join(output_dir, video_name)

cap = cv2.VideoCapture(args.video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare to track max and min y-coordinates and heights of crash test sign
max_y = float("-inf")
min_y = float("inf")
desired_sign_height = 40  # Desired height of the crash test sign in pixels
heights = []  # List to store the heights of the crash test sign in each frame

# First pass to calculate average height of crash test sign
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection on the frame
    results = model(frame, verbose=False)
    detections = results[0].boxes  # Access detected bounding boxes

    # Get crash test sign class ID
    crash_test_sign_id = next(
        key for key, value in model.names.items() if value == "crash_test_sign"
    )
    crash_test_signs = [box for box in detections if box.cls == crash_test_sign_id]

    if crash_test_signs:
        # Define tracking criterion (e.g., closest to the center)
        frame_center_x, frame_center_y = frame.shape[1] / 2, frame.shape[0] / 2

        # Find the object closest to the center
        closest_sign = min(
            crash_test_signs,
            key=lambda box: (
                (box.xywh[0][0] - frame_center_x) ** 2
                + (box.xywh[0][1] - frame_center_y) ** 2
            ).sum(),
        )

        # Extract y-coordinates and calculate height
        y1, y2 = int(closest_sign.xyxy[0][1]), int(closest_sign.xyxy[0][3])
        height_of_sign = y2 - y1
        heights.append(height_of_sign)  # Add height to list

        # Update min_y and max_y based on tracked position
        min_y = min(min_y, y1)
        max_y = max(max_y, y2)

# Calculate average height of the crash test sign
average_height = sum(heights) / len(heights) if heights else 1  # Avoid division by zero
scaling_factor = desired_sign_height / average_height

# Release video resources for the first pass
cap.release()

# Re-open video for cropping and saving
cap = cv2.VideoCapture(args.video_path)

# Initialize video writer with the same FPS and cropped dimensions
output_width = original_width
output_height = int((max_y - min_y) * scaling_factor)  # Scale the crop height
out = cv2.VideoWriter(
    output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width, output_height)
)

# Second pass to crop, resize, and save frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Crop the frame vertically from min_y to max_y
    cropped_frame = frame[min_y:max_y, :]

    # Resize the cropped frame based on the calculated scaling factor
    resized_cropped_frame = cv2.resize(cropped_frame, (output_width, output_height))

    # Write the resized frame to output
    out.write(resized_cropped_frame)

# Release resources
cap.release()
out.release()
print(f"Cropped video saved at {output_path}")

import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(
    description="YOLO object tracking on a video.")
parser.add_argument(
    "video_path",
    type=str,
    help="Path to the video file to be processed",
)
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

    # Calculate the new width to maintain aspect ratio with height of 640
    original_height, original_width = frame.shape[:2]
    new_height = 640
    new_width = int(original_width * (new_height / original_height))

    # Resize the frame with the new dimensions
    frame = cv2.resize(frame, (new_width, new_height))
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

        # Draw the bounding box around the tracked object
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "Tracking",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # Track the vertical center of the bounding box
        center_y = (y1 + y2) // 2
        y_positions.append(center_y)
        frame_times.append(frame_time)

    # Show the frame with bounding box
    cv2.imshow("YOLO Tracking", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Update frame time based on FPS
    frame_time += 1 / fps

# Release video resources and close display windows
cap.release()

frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
cv2.destroyAllWindows()

# Calculate the mean of y_positions and shift values to center around zero
mean_y = sum(y_positions) / len(y_positions)
centered_y_positions = [y - mean_y for y in y_positions]

inverted_y_positions = [frame_height - y for y in y_positions]
# Plot y-coordinate over time, centered around zero
plt.figure(figsize=(10, 5))
plt.plot(
    frame_times,
    inverted_y_positions,
    label="Centered Vertical Position (y)",
    color="blue",
)
plt.xlabel("Time (s)")
plt.ylabel("Centered Vertical Position (y-coordinate)")
plt.title("Centered Oscillating Motion of Selected Object Over Time")
plt.legend()
plt.grid(True)
plt.show()

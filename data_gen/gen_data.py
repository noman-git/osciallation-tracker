import cv2
import random
import os
from tqdm import tqdm  # Import tqdm for progress bar

curr_dir = os.getcwd()
project_dir = os.path.join(curr_dir, "..", "datasets", "crashtest_dataset")

# Define directories within 'obj_detection/yolo_dataset'
crash_test_dir = os.path.join(curr_dir, "crash_test")
background_dir = os.path.join(curr_dir, "backgrounds")
output_dir = os.path.join(project_dir, "images/train")
label_dir = os.path.join(project_dir, "labels/train")
verified_dir = os.path.join(project_dir, "verified_images")
val_output_dir = os.path.join(project_dir, "images/val")
val_label_dir = os.path.join(project_dir, "labels/val")
val_verified_dir = os.path.join(project_dir, "verified_images/val")

# Create output directories if they don’t exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)
os.makedirs(verified_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)
os.makedirs(val_verified_dir, exist_ok=True)


def overlay_image(background, overlay, position):
    h, w = overlay.shape[:2]
    y1, y2 = position[1], position[1] + h
    x1, x2 = position[0], position[0] + w

    if y2 > background.shape[0] or x2 > background.shape[1]:
        return background

    overlay_mask = overlay[:, :, 3] / 255.0
    for c in range(0, 3):
        background[y1:y2, x1:x2,
                   c] = (overlay_mask * overlay[:, :, c] +
                         (1 - overlay_mask) * background[y1:y2, x1:x2, c])
    return background


def random_scale(image):
    scale_factor = random.uniform(0.4, 0.8)
    new_size = (int(image.shape[1] * scale_factor),
                int(image.shape[0] * scale_factor))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)


# Standard background size
standard_size = (640, 640)

# Load data
crash_test_images = [
    f for f in os.listdir(crash_test_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))
]
background_images = [
    f for f in os.listdir(background_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))
]

# Generate images with progress tracking
for i, bg_image_name in tqdm(enumerate(background_images),
                             total=len(background_images)):
    background_image = cv2.imread(os.path.join(background_dir, bg_image_name))
    background_image = cv2.resize(background_image, standard_size)

    # Randomly select a crash test image
    crash_test_image_name = random.choice(crash_test_images)
    crash_test_image_path = os.path.join(crash_test_dir, crash_test_image_name)
    crash_test_image = cv2.imread(crash_test_image_path, cv2.IMREAD_UNCHANGED)

    # Check if the image was loaded successfully
    if crash_test_image is None:
        print(
            f"Warning: Failed to load image {crash_test_image_path}. Skipping."
        )
        continue  # Skip to the next iteration if loading failed

    # Apply scaling transformation for 80% of images
    if random.random() < 0.8:
        crash_test_image = random_scale(crash_test_image)

    # Random position for overlay
    h, w = background_image.shape[:2]
    x = random.randint(0, max(0, w - crash_test_image.shape[1]))
    y = random.randint(0, max(0, h - crash_test_image.shape[0]))

    # Overlay the crash test image on the background image
    combined_image = overlay_image(background_image.copy(), crash_test_image,
                                   (x, y))
    verified_image = combined_image.copy()

    # Calculate bounding box coordinates around the overlayed image
    min_x, min_y = x, y
    max_x, max_y = x + crash_test_image.shape[1], y + crash_test_image.shape[0]

    # Calculate normalized YOLO bounding box coordinates
    box_center_x = ((min_x + max_x) / 2) / w
    box_center_y = ((min_y + max_y) / 2) / h
    box_width = (max_x - min_x) / w
    box_height = (max_y - min_y) / h

    # Write annotation file (80% for training, 20% for validation)
    if random.random() < 0.8:
        label_path = os.path.join(label_dir, f"combined_{i}.txt")
        output_image_path = os.path.join(output_dir, f"combined_{i}.jpg")
        verified_path = os.path.join(verified_dir,
                                     f"combined_{i}_verified.jpg")
    else:
        label_path = os.path.join(val_label_dir, f"combined_{i}.txt")
        output_image_path = os.path.join(val_output_dir, f"combined_{i}.jpg")
        verified_path = os.path.join(val_verified_dir,
                                     f"combined_{i}_verified.jpg")

    with open(label_path, "w") as f:
        f.write(f"0 {box_center_x} {box_center_y} {box_width} {box_height}\n")

    # Draw bounding box on the verified image
    cv2.rectangle(verified_image, (min_x, min_y), (max_x, max_y), (0, 255, 0),
                  2)

    # Save the combined image and the verified image
    cv2.imwrite(output_image_path, combined_image)
    cv2.imwrite(verified_path, verified_image)

print(
    "YOLO training images, labels, and verification images created successfully."
)

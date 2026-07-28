import os
import random
import shutil

# Dataset folders
BASE_DIR = "dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")

# Split percentage
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

random.seed(42)

# Classes
classes = ["healthy", "lumpy"]

for cls in classes:
    train_path = os.path.join(TRAIN_DIR, cls)
    val_path = os.path.join(VAL_DIR, cls)
    test_path = os.path.join(TEST_DIR, cls)

    # Create folders if they don't exist
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # Read images
    images = [
        img for img in os.listdir(train_path)
        if img.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    total = len(images)
    val_count = int(total * VAL_SPLIT)
    test_count = int(total * TEST_SPLIT)

    val_images = images[:val_count]
    test_images = images[val_count:val_count + test_count]

    # Copy to Validation
    for img in val_images:
        shutil.copy(
            os.path.join(train_path, img),
            os.path.join(val_path, img)
        )

    # Copy to Test
    for img in test_images:
        shutil.copy(
            os.path.join(train_path, img),
            os.path.join(test_path, img)
        )

    print(f"{cls}")
    print(f"Total Images : {total}")
    print(f"Validation   : {len(val_images)}")
    print(f"Test         : {len(test_images)}")
    print("-" * 30)

print("✅ Dataset Split Completed Successfully!")
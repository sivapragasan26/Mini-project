import os
import random
import shutil

# Paths
original_dataset = r"C:\Users\somanathan\pulmo-vision\dataset"
reduced_dataset = r"C:\Users\somanathan\pulmo-vision\dataset_small"

# Number of images per class
N = 2000  

os.makedirs(reduced_dataset, exist_ok=True)

for cls in os.listdir(original_dataset):
    cls_path = os.path.join(original_dataset, cls)
    if not os.path.isdir(cls_path):
        continue  # skip non-folder files

    save_path = os.path.join(reduced_dataset, cls)
    os.makedirs(save_path, exist_ok=True)

    images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tif','.tiff'))]
    random.shuffle(images)

    selected = images[:min(N, len(images))]  # take up to N images

    for img in selected:
        shutil.copy(os.path.join(cls_path, img), save_path)

    print(f"✅ {cls}: {len(selected)} images copied")

print("🎉 Reduced dataset created at:", reduced_dataset)

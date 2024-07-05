import os
import shutil

train_data_dir = "landcover.ai.v1/train/image"
train_label_dir = "landcover.ai.v1/train/label"

val_data_dir = "landcover.ai.v1/val/image"
val_label_dir = "landcover.ai.v1/val/label"

test_data_dir = "landcover.ai.v1/test/image"
test_label_dir = "landcover.ai.v1/test/label"

paths = [
    train_data_dir,
    train_label_dir,
    val_data_dir,
    val_label_dir,
    test_data_dir,
    test_label_dir,
]

for p in paths:
    if not os.path.exists(p):
        os.makedirs(p)
        print(f"Folder '{p}' created.")
    else:
        print(f"Folder '{p}' already exists.")


with open("landcover.ai.v1/train.txt") as f:
    train_split = f.read().splitlines()

with open("landcover.ai.v1/val.txt") as f:
    val_split = f.read().splitlines()

with open("landcover.ai.v1/test.txt") as f:
    test_split = f.read().splitlines()

print("Train, Val, Test splits count")
print(len(train_split), len(val_split), len(test_split))
print("Last item of each split")
print(train_split[-1], val_split[-1], test_split[-1])

for i in train_split:
    source_file_img = os.path.join("landcover.ai.v1/output", f"{i}.tif")
    source_file_label = os.path.join("landcover.ai.v1/output", f"{i}_m.tif")
    destination_file_img = os.path.join(train_data_dir, f"{i}.tif")
    destination_file_label = os.path.join(train_label_dir, f"{i}.tif")
    if os.path.isfile(source_file_img) and not os.path.exists(destination_file_img):
        shutil.copy2(source_file_img, destination_file_img)
    if os.path.isfile(source_file_label) and not os.path.exists(destination_file_label):
        shutil.copy2(source_file_label, destination_file_label)

for i in val_split:
    source_file_img = os.path.join("landcover.ai.v1/output", f"{i}.tif")
    source_file_label = os.path.join("landcover.ai.v1/output", f"{i}_m.tif")
    destination_file_img = os.path.join(val_data_dir, f"{i}.tif")
    destination_file_label = os.path.join(val_label_dir, f"{i}.tif")
    if os.path.isfile(source_file_img) and not os.path.exists(destination_file_img):
        shutil.copy2(source_file_img, destination_file_img)
    if os.path.isfile(source_file_label) and not os.path.exists(destination_file_label):
        shutil.copy2(source_file_label, destination_file_label)

for i in test_split:
    source_file_img = os.path.join("landcover.ai.v1/output", f"{i}.tif")
    source_file_label = os.path.join("landcover.ai.v1/output", f"{i}_m.tif")
    destination_file_img = os.path.join(test_data_dir, f"{i}.tif")
    destination_file_label = os.path.join(test_label_dir, f"{i}.tif")
    if os.path.isfile(source_file_img) and not os.path.exists(destination_file_img):
        shutil.copy2(source_file_img, destination_file_img)
    if os.path.isfile(source_file_label) and not os.path.exists(destination_file_label):
        shutil.copy2(source_file_label, destination_file_label)

print("File copying completed.")

import os
from PIL import Image
import numpy as np
import albumentations as A

if not os.path.exists("./landcover.ai.v1/augmented"):
    os.makedirs("./landcover.ai.v1/augmented/image")
    os.makedirs("./landcover.ai.v1/augmented/label")
    print(f"Folders created.")

train_data_dir = "landcover.ai.v1/train/image"
train_label_dir = "landcover.ai.v1/train/label"

with open("landcover.ai.v1/train.txt") as f:
    train_split = f.read().splitlines()

for s in train_split:
    pillow_image = Image.open(f"{train_data_dir}/{s}.tif")
    image = np.array(pillow_image)

    pillow_mask = Image.open(f"{train_label_dir}/{s}.tif")
    mask = np.array(pillow_mask)

    transform = A.Compose([
        A.OneOf([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.HueSaturationValue(p=1),
            A.ToGray(p=1),
            A.RandomBrightnessContrast(p=1)
        ], p=1)
    ])

    transformed = transform(image=image, mask=mask)
    transformed_image = transformed['image']
    transformed_mask = transformed['mask']

    pil_transformed_image = Image.fromarray(transformed_image)
    pil_transformed_mask = Image.fromarray(transformed_mask)

    pil_transformed_image.save(f"./landcover.ai.v1/augmented/image/{s}_AUG.tif")
    pil_transformed_mask.save(f"./landcover.ai.v1/augmented/label/{s}_AUG.tif")
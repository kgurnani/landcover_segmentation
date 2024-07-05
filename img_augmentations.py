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

    hflip = A.HorizontalFlip(p=1)
    vflip = A.VerticalFlip(p=1)
    hue = A.HueSaturationValue(p=1)
    gray = A.ToGray(p=1)
    brightness = A.RandomBrightnessContrast(p=1)
    ### apply hflip
    transformed = hflip(image=image, mask=mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]

    pil_transformed_image = Image.fromarray(transformed_image)
    pil_transformed_mask = Image.fromarray(transformed_mask)

    pil_transformed_image.save(f"./landcover.ai.v1/augmented/image/{s}_hflip.tif")
    pil_transformed_mask.save(f"./landcover.ai.v1/augmented/label/{s}_hflip.tif")
    ###apply vflip
    transformed = vflip(image=image, mask=mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]

    pil_transformed_image = Image.fromarray(transformed_image)
    pil_transformed_mask = Image.fromarray(transformed_mask)

    pil_transformed_image.save(f"./landcover.ai.v1/augmented/image/{s}_vflip.tif")
    pil_transformed_mask.save(f"./landcover.ai.v1/augmented/label/{s}_vflip.tif")
    ### apply hue
    transformed = hue(image=image, mask=mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]

    pil_transformed_image = Image.fromarray(transformed_image)
    pil_transformed_mask = Image.fromarray(transformed_mask)

    pil_transformed_image.save(f"./landcover.ai.v1/augmented/image/{s}_hue.tif")
    pil_transformed_mask.save(f"./landcover.ai.v1/augmented/label/{s}_hue.tif")
    ### apply gray
    transformed = gray(image=image, mask=mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]

    pil_transformed_image = Image.fromarray(transformed_image)
    pil_transformed_mask = Image.fromarray(transformed_mask)

    pil_transformed_image.save(f"./landcover.ai.v1/augmented/image/{s}_gray.tif")
    pil_transformed_mask.save(f"./landcover.ai.v1/augmented/label/{s}_gray.tif")
    ### apply brightness
    transformed = brightness(image=image, mask=mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]

    pil_transformed_image = Image.fromarray(transformed_image)
    pil_transformed_mask = Image.fromarray(transformed_mask)

    pil_transformed_image.save(f"./landcover.ai.v1/augmented/image/{s}_bright.tif")
    pil_transformed_mask.save(f"./landcover.ai.v1/augmented/label/{s}_bright.tif")

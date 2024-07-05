import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from PIL import Image
import glob
import os


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerUNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNet
        for feature in features:
            self.downs.append(self._double_conv(in_channels, feature))
            in_channels = feature

        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(self._double_conv(feature * 2, feature))

        self.bottleneck = self._double_conv(features[-1], features[-1] * 2)
        self.transformer = TransformerBlock(features[-1] * 2, num_heads=8)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        skip_connections = []

        # Down part
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Apply transformer to the bottleneck
        b, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
        x = self.transformer(x)
        x = x.permute(1, 2, 0).view(b, c, h, w)

        # Up part
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


# Define the loss function
class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, predictions, targets):
        predictions = torch.softmax(predictions, dim=1)

        dice_loss = 0
        for i in range(self.num_classes):
            pred_class = predictions[:, i, :, :]
            target_class = (targets == i).float()

            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum()

            dice_loss += 1 - (2.0 * intersection + self.smooth) / (union + self.smooth)

        return dice_loss / self.num_classes


def create_color_map():
    return {
        (0, 0, 0): 0,
        (1, 1, 1): 1,
        (2, 2, 2): 2,
        (3, 3, 3): 3,
        (4, 4, 4): 4,
    }


# Data preprocessing
def preprocess_data(image, mask):
    image = transforms.ToTensor()(image)
    mask = np.array(mask)
    color_to_class = create_color_map()
    mask_classes = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
    for color, class_id in color_to_class.items():
        mask_classes[(mask == color).all(axis=2)] = class_id
    mask_classes = torch.from_numpy(mask_classes)
    return image, mask_classes


# Training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for images, masks in tqdm(train_loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    val_iou = 0

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            val_iou += iou_score(outputs, masks)

    return total_loss / len(val_loader), val_iou / len(val_loader)


# IoU metric
def iou_score(output, target):
    smooth = 1e-5
    output = torch.argmax(output, dim=1)
    intersection = (output == target).float().sum((1, 2))
    union = output.float().sum((1, 2)) + target.float().sum((1, 2)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


# Custom dataset
class CustomDataset(Dataset):
    def __init__(self, image_paths, target_paths, preprocess_fn):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.preprocess_fn = preprocess_fn

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.target_paths[index])
        return self.preprocess_fn(image, mask)

    def __len__(self):
        return len(self.image_paths)


# Main training loop
def main():
    # Hyperparameters
    batch_size = 8
    num_epochs = 1
    learning_rate = 1e-4
    num_classes = 5  # Including background
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_img_paths = glob.glob(os.path.join("landcover.ai.v1/val/image", "*.tif"))
    train_img_paths = sorted(train_img_paths)

    train_mask_paths = glob.glob(os.path.join("landcover.ai.v1/val/label", "*.tif"))
    train_mask_paths = sorted(train_mask_paths)

    val_img_paths = glob.glob(os.path.join("landcover.ai.v1/val/image", "*.tif"))
    val_img_paths = sorted(val_img_paths)

    val_mask_paths = glob.glob(os.path.join("landcover.ai.v1/val/label", "*.tif"))
    val_mask_paths = sorted(val_mask_paths)

    train_dataset = CustomDataset(
        image_paths=train_img_paths,
        target_paths=train_mask_paths,
        preprocess_fn=preprocess_data,
    )
    val_dataset = CustomDataset(
        image_paths=val_img_paths,
        target_paths=val_mask_paths,
        preprocess_fn=preprocess_data,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Initialize model, loss, and optimizer
    model = TransformerUNet(in_channels=3, out_channels=num_classes).to(device)
    criterion = DiceLoss(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_iou = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation IoU: {val_iou:.4f}")
        print("-----------------------------")

    print("Training completed!")


if __name__ == "__main__":
    main()

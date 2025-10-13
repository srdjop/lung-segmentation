import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
import segmentation_models_pytorch as smp
import cv2
import numpy as np
from torch.utils.data import Subset # Dodaj na vrh

from data_loader import LungSegmentationDataset, get_train_transforms, get_val_transforms

# --- Loss function and metrics ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice_score = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice_score

def dice_score(logits, targets, smooth=1.0):
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        probs = (probs > 0.5).float()
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        score = (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
    return score.item()

# --- Training and Validation function ---
def train_one_epoch(loader, model, optimizer, loss_fn, device):
    model.train()
    loop = tqdm(loader, desc="Training")
    total_loss = 0.0

    for data, targets, _ in loop: 
        data = data.to(device)
        targets = targets.to(device).unsqueeze(1)
        
        # Forward pass
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)

def evaluate(loader, model, loss_fn, device):
    model.eval()
    total_loss, total_dice = 0.0, 0.0
    images_to_log = []

    with torch.no_grad():
        for i, (data, targets, _) in enumerate(tqdm(loader, desc="Validation")):
            data = data.to(device)
            targets = targets.to(device).unsqueeze(1)

            predictions = model(data)
            loss = loss_fn(predictions, targets)
            dice = dice_score(predictions, targets)

            total_loss += loss.item()
            total_dice += dice
            
            if i == 0: # Just first batch for visualization
                img_tensor = data[0]
                true_mask_tensor = targets[0]
                pred_mask_tensor = (torch.sigmoid(predictions[0]) > 0.5).float()
                
                img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                true_mask_np = (true_mask_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
                pred_mask_np = (pred_mask_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
                
                images_to_log = [
                    wandb.Image(img_np, caption="Input Image"),
                    wandb.Image(true_mask_np, caption="Ground Truth Mask"),
                    wandb.Image(pred_mask_np, caption="Predicted Mask")
                ]

    avg_loss, avg_dice = total_loss / len(loader), total_dice / len(loader)
    print(f"Validation -> Avg Loss: {avg_loss:.4f}, Avg Dice Score: {avg_dice:.4f}")
    return avg_loss, avg_dice, images_to_log

# --- Main function ---
def main():
    # --- Hyperparameters ---
    IMG_SIZE = 512
    BATCH_SIZE = 4 # Smaller batch size for less data
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ENCODER = "resnet34"

    wandb.init(
        project="lung-segmentation-ams-unet-sanity-check",
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": f"U-Net with {ENCODER} (Sanity Check)",
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "image_size": IMG_SIZE,
        }
    )

    # --- Loading the data ---
    transforms = get_val_transforms(IMG_SIZE)
    full_dataset = LungSegmentationDataset(image_dir='data/images', mask_dir='data/masks', transform=transforms)
    
    # Small subset to force the overfitting
    sanity_dataset = Subset(full_dataset, range(16))
    
    train_loader = DataLoader(sanity_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # --- Model, Loss, Optimizator ---
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(DEVICE)
    loss_fn = DiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, DEVICE)
        wandb.log({"train_loss": train_loss, "epoch": epoch + 1})
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}")

    wandb.finish()

    print("Training finished.")

if __name__ == '__main__':
    main()
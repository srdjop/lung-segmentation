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

from data_loader import LungSegmentationDataset, get_train_transforms, get_val_transforms

# --- Loss Function and Metrics---
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

# --- Training and Validation ---
def train_one_epoch(loader, model, optimizer, loss_fn, device):
    model.train()
    loop = tqdm(loader, desc="Training")
    total_loss = 0.0
    total_dice = 0.0

    for data, targets, _ in loop: 
        data = data.to(device)
        targets = targets.to(device).unsqueeze(1)
        
        # Forward pass
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        #  Dice Score for training batch 
        dice = dice_score(predictions, targets)
        total_dice += dice
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    return avg_loss, avg_dice 

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
            
            if i == 0: # Logging just the first batch for visualization
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

# --- Main Function ---
def main():
    # --- Hyperparameters ---
    IMG_SIZE = 512       # U-Net works perfect on 512x512, indicating the faster model training
    BATCH_SIZE = 8       # Bigger batch size
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 50
    VAL_SPLIT = 0.2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ENCODER = "resnet34" # Fast and robust encoder

    # --- wandb Initialization ---
    wandb.init(
        project="lung-segmentation-extended-dataset", 
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": f"U-Net with {ENCODER}",
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "image_size": IMG_SIZE,
            "dataset": "Original + Kaggle"
        }
    )

    # --- Loading the data ---
    # New transformation for 512x512
    train_transforms = get_train_transforms(IMG_SIZE)
    val_transforms = get_val_transforms(IMG_SIZE)
    
    full_dataset = LungSegmentationDataset(image_dir='data/images', mask_dir='data/masks')
    train_size = int(len(full_dataset) * (1 - VAL_SPLIT))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_ds.dataset.transform = train_transforms
    val_ds.dataset.transform = val_transforms
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # --- Model, Loss, Optimizator ---
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(DEVICE)
    
    loss_fn = DiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    wandb.watch(model, log="all")
    
    best_val_dice = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_loss, train_dice = train_one_epoch(train_loader, model, optimizer, loss_fn, DEVICE)
        val_loss, val_dice, viz_images = evaluate(val_loader, model, loss_fn, DEVICE)
        
        wandb.log({
            "train_loss": train_loss,
            "train_dice_score": train_dice, 
            "val_loss": val_loss,
            "val_dice_score": val_dice,
            "epoch": epoch + 1,
            "predictions": viz_images
        })
        
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), "best_unet_model_ext_dataset.pth")
            print(f"=> Saved new best model with Dice Score: {val_dice:.4f}")

    wandb.finish()
    print("Training finished.")

if __name__ == '__main__':
    main()
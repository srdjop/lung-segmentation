import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
from transformers import SamModel
import cv2
import numpy as np

# Importing the Data Loader
from data_loader import LungSegmentationDataset, get_train_transforms, get_val_transforms

# --- Model SAM ---
class SAM_FineTune(nn.Module):
    def __init__(self, model_name="facebook/sam-vit-base"):
        super(SAM_FineTune, self).__init__()
        self.model = SamModel.from_pretrained(model_name)

    def forward(self, pixel_values, input_boxes=None):
        outputs = self.model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False, return_dict=False)
        return outputs[0]

# --- Loss Function ---
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

class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=0.9, weight_bce=0.1):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce

    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        bce = self.bce_loss(logits, targets)
        return (self.weight_dice * dice) + (self.weight_bce * bce)

# --- Metrics ---
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

    for data, targets, bboxes in loop:
        data, targets, bboxes = data.to(device), targets.to(device).unsqueeze(1), bboxes.to(device)
        
        predictions = model(pixel_values=data, input_boxes=bboxes.unsqueeze(1))
        
        if predictions.ndim == 3: predictions = predictions.unsqueeze(1)
        
        upscaled_predictions = nn.functional.interpolate(predictions, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        
        loss = loss_fn(upscaled_predictions, targets)
        
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
        for i, (data, targets, bboxes) in enumerate(tqdm(loader, desc="Validation")):
            data, targets, bboxes = data.to(device), targets.to(device).unsqueeze(1), bboxes.to(device)

            predictions = model(pixel_values=data, input_boxes=bboxes.unsqueeze(1))
            if predictions.ndim == 3: predictions = predictions.unsqueeze(1)
            
            upscaled_predictions = nn.functional.interpolate(predictions, size=targets.shape[-2:], mode="bilinear", align_corners=False)
            
            loss = loss_fn(upscaled_predictions, targets)
            dice = dice_score(upscaled_predictions, targets)

            total_loss += loss.item()
            total_dice += dice
            
            if i == 0:
                img_tensor, true_mask_tensor, pred_mask_tensor = data[0], targets[0], (torch.sigmoid(upscaled_predictions[0]) > 0.5).float()
                bbox = bboxes[0].cpu().numpy().astype(int)
                img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                img_with_bbox = cv2.rectangle(img_np.copy(), (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                true_mask_np = (true_mask_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
                pred_mask_np = (pred_mask_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
                images_to_log = [wandb.Image(img_with_bbox, caption="Input Image with BBox"), wandb.Image(true_mask_np, caption="Ground Truth"), wandb.Image(pred_mask_np, caption="Prediction")]

    avg_loss, avg_dice = total_loss / len(loader), total_dice / len(loader)
    print(f"Validation -> Avg Loss: {avg_loss:.4f}, Avg Dice Score: {avg_dice:.4f}")
    return avg_loss, avg_dice, images_to_log

# --- Main Function ---
def main():
    # --- Hyperparameters ---
    IMG_SIZE = 1024
    BATCH_SIZE = 2
    VAL_SPLIT = 0.2
    MODEL_NAME = "facebook/sam-vit-base"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Phase 1: Only Decoder Training ---
    PHASE1_EPOCHS = 5
    PHASE1_LR = 1e-4

    # --- Phase 2: Fine-tuning the whole system ---
    PHASE2_EPOCHS = 15
    PHASE2_LR_DECODER = 1e-5
    PHASE2_LR_ENCODER = 1e-6
    UNFREEZE_BLOCKS = 4

    # --- wandb initialization ---
    wandb.init(project="lung-segmentation-ams-robust")

    # --- Loading the data ---
    train_transforms = get_train_transforms(IMG_SIZE)
    val_transforms = get_val_transforms(IMG_SIZE)
    full_dataset = LungSegmentationDataset(image_dir='data/images', mask_dir='data/masks', transform=train_transforms)
    train_size = int(len(full_dataset) * (1 - VAL_SPLIT))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset.transform = val_transforms 
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # --- Phase 1: Decoder Training ---
    print("\n--- Phase 1 starting: Decoder ---")
    model = SAM_FineTune(model_name=MODEL_NAME).to(DEVICE)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("model.mask_decoder")
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=PHASE1_LR)
    loss_fn = CombinedLoss()
    
    for epoch in range(PHASE1_EPOCHS):
        print(f"\n--- Phase 1, Epochs {epoch+1}/{PHASE1_EPOCHS} ---")
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, DEVICE)
        val_loss, val_dice, _ = evaluate(val_loader, model, loss_fn, DEVICE)
        wandb.log({"phase1_train_loss": train_loss, "phase1_val_loss": val_loss, "phase1_val_dice": val_dice, "epoch": epoch})

    # --- Faza 2: Fino Podešavanje ---
    print("\n--- Phase 2 starting: Fine-tuning encoder and decoder ---")
    unfreeze_layers = [f"layers.{i}" for i in range(12 - UNFREEZE_BLOCKS, 12)]
    for name, param in model.named_parameters():
        if name.startswith("model.vision_encoder") and any(layer in name for layer in unfreeze_layers):
            param.requires_grad = True

    decoder_params = list(model.model.mask_decoder.parameters())
    encoder_params = [p for p in model.parameters() if p.requires_grad and not any(id(p) == id(dp) for dp in decoder_params)]
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': PHASE2_LR_ENCODER},
        {'params': decoder_params, 'lr': PHASE2_LR_DECODER}
    ])
    
    best_val_dice = 0.0
    for epoch in range(PHASE2_EPOCHS):
        print(f"\n--- Faza 2, Epoha {epoch+1}/{PHASE2_EPOCHS} ---")
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, DEVICE)
        val_loss, val_dice, viz_images = evaluate(val_loader, model, loss_fn, DEVICE)
        wandb.log({
            "phase2_train_loss": train_loss, "phase2_val_loss": val_loss, "phase2_val_dice": val_dice,
            "epoch": PHASE1_EPOCHS + epoch, "predictions": viz_images
        })
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), "best_sam2_model.pth")
            print(f"=> Saved the best model with Dice Score: {val_dice:.4f}")

    wandb.finish()
    print("Training finished.")

if __name__ == '__main__':
    main()
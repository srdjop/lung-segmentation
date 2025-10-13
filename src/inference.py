import os
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import random
from huggingface_hub import hf_hub_download 


# --- Model Definition ---
def get_model(encoder_name="resnet34", in_channels=3, classes=1):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None, # We don't need pretrained weights, we'll load our
        in_channels=in_channels,
        classes=classes,
    )
    return model

# --- Image transformation ---
def get_inference_transforms(img_size):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ])

# --- Main Inference Function ---
def segment_lung(model, image_path, img_size, device):
    """
    Accepting the model and path to the image, returning binary mask and original image.
    """
    image_orig = Image.open(image_path).convert("RGB")
    image_np = np.array(image_orig)
    
    transforms = get_inference_transforms(img_size)
    augmented = transforms(image=image_np)
    image_tensor = augmented['image'].unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        pred_mask = (probs > 0.5).float().squeeze().cpu().numpy()

    return pred_mask, image_np

# --- Visualization function ---
def visualize_prediction(original_image, predicted_mask, save_path=None):
    """
    Making the picture where predicted mask is colored above the original image.
    """
    original_height, original_width = original_image.shape[:2]
    
    # INTER_NEAREST best suitable for interpolation for masks
    mask_resized = cv2.resize(
        predicted_mask, 
        (original_width, original_height), 
        interpolation=cv2.INTER_NEAREST
    )
    # ==========================================================

    mask_colored = np.zeros_like(original_image, dtype=np.uint8)
    mask_colored[mask_resized == 1] = [0, 255, 0] # Green color for lungs
    
    # Overlaying mask
    overlayed_image = cv2.addWeighted(original_image, 1, mask_colored, 0.4, 0)
    
    if save_path:
        Image.fromarray(overlayed_image).save(save_path)
        print(f"Visualization saved in: {save_path}")
        
    return overlayed_image

if __name__ == '__main__':
    # --- Configuration, loading model from Hugging-Face Hub ---
    HF_REPO_ID = "srdjoo14/lung-segmentation-unet-resnet34" 
    HF_FILENAME = "best_unet_model.pth"
    
    IMG_SIZE = 512
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    IMAGE_DIR = 'data/images'
    
    # --- Loading the model ---
    print(f"Downloading the model '{HF_FILENAME}' from Hugging Face Hub repository: {HF_REPO_ID}")
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    print(f"Model downloaded on local path: {model_path}")
    
    model = get_model().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    print("Model succesfully loaded.")
    # ============================================================
    
    # Choosing the random image from dataset
    test_images = os.listdir(IMAGE_DIR)
    if not test_images:
        print(f"There is no image in the folder: {IMAGE_DIR}")
    else:
        random_image_name = random.choice(test_images)
        image_path_to_test = os.path.join(IMAGE_DIR, random_image_name)
        
        print(f"\nTesting image: {image_path_to_test}")
        
        # 1. Prediction 
        predicted_mask_np, original_image_np = segment_lung(
            model=model,
            image_path=image_path_to_test,
            img_size=IMG_SIZE,
            device=DEVICE
        )
        
        # 2. Visualization and saving the results
        os.makedirs("outputs", exist_ok=True)
        save_filename = f"outputs/predicted_{random_image_name}"
        
        visualize_prediction(
            original_image=original_image_np,
            predicted_mask=predicted_mask_np,
            save_path=save_filename
        )
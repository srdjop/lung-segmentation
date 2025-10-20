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
# We must define the same architecture as during training
def get_model(encoder_name="resnet34", in_channels=3, classes=1):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,  # We don't need pre-trained weights, we will load our own
        in_channels=in_channels,
        classes=classes,
    )
    return model

# --- Image Transformations ---
# We use the same transformations as for the validation set, WITHOUT augmentations
def get_inference_transforms(img_size):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ])

# --- Main Inference Function ---
def segment_lung(model, image_path, img_size, device):
    """
    Takes a model and an image path, returns a binary mask (NumPy array) and the original image.
    """
    image_orig = Image.open(image_path).convert("RGB")
    image_np = np.array(image_orig)
    
    transforms = get_inference_transforms(img_size)
    augmented = transforms(image=image_np)
    image_tensor = augmented['image'].unsqueeze(0).to(device)  # Add batch dimension
    
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Prediction
        logits = model(image_tensor)
        
        # Process the output
        probs = torch.sigmoid(logits)
        # The mask is returned as a NumPy array with values of 0.0 or 1.0
        pred_mask_np = (probs > 0.5).float().squeeze().cpu().numpy()

    return pred_mask_np, image_np

# --- Visualization Function ---
def visualize_prediction(original_image, predicted_mask, save_path=None):
    """
    Creates an image where the predicted mask is overlaid in color on the original image.
    """
    original_height, original_width = original_image.shape[:2]
    
    # Resize the mask back to the original image size
    # cv2.INTER_NEAREST is the best interpolation for masks to avoid blurry edges
    mask_resized = cv2.resize(
        predicted_mask, 
        (original_width, original_height), 
        interpolation=cv2.INTER_NEAREST
    )
    
    # Convert the mask to a displayable format (3 channels, uint8)
    mask_colored = np.zeros_like(original_image, dtype=np.uint8)
    mask_colored[mask_resized == 1] = [0, 255, 0]  # Green color for the lungs
    
    # Overlay the mask onto the original image
    overlayed_image = cv2.addWeighted(original_image, 1, mask_colored, 0.4, 0)
    
    if save_path:
        Image.fromarray(overlayed_image).save(save_path)
        print(f"Visualization saved to: {save_path}")
        
    return overlayed_image

# --- Demonstration ---
if __name__ == '__main__':
    # --- Configuration ---
    HF_REPO_ID = "srdjoo14/lung-segmentation-unet-resnet34" 
    HF_FILENAME = "best_unet_model_ext_dataset.pth" 
    IMG_SIZE = 512
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    #IMAGE_DIR = 'data/images'
    IMAGE_PATH_TO_TEST = 'test_images/test_slika.png'
    
    # --- Load Model from Hugging Face Hub ---
    print(f"Downloading model from Hugging Face Hub: {HF_REPO_ID}")
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, revision="adcf431f68b023c541660f95beb07fced46cd0ed")
    model = get_model().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    print("Model loaded successfully.")
    
    # --- Test on a Random Image ---
    # test_images = os.listdir(IMAGE_DIR)
    # if not test_images:
    #     print(f"No images found in directory: {IMAGE_DIR}")
    # else:
    #     random_image_name = random.choice(test_images)
    #     image_path_to_test = os.path.join(IMAGE_DIR, random_image_name)
        
    #     print(f"\nTesting on image: {image_path_to_test}")
        
    #     # 1. Get the prediction from the main function
    #     #    predicted_mask_np is the binary image (an array with 0s and 1s)
    #     predicted_mask_np, original_image_np = segment_lung(
    #         model=model,
    #         image_path=image_path_to_test,
    #         img_size=IMG_SIZE,
    #         device=DEVICE
    #     )
    #    
    #    # Create the 'outputs' directory if it doesn't exist
    #   
        # --- Test on the Specified Image ---
    if not os.path.exists(IMAGE_PATH_TO_TEST):
        print(f"ERROR: Image not found at path: {IMAGE_PATH_TO_TEST}")
        print("Please make sure you have created the 'test_images' folder and placed the image inside.")
    else:
        print(f"\nTesting on image: {IMAGE_PATH_TO_TEST}")
        
        # 1. Get the prediction from the main function
        predicted_mask_np, original_image_np = segment_lung(
            model=model,
            image_path=IMAGE_PATH_TO_TEST, # Koristimo specifičnu putanju
            img_size=IMG_SIZE,
            device=DEVICE
        )

        os.makedirs("outputs", exist_ok=True)
        
        # Dobijamo čisto ime fajla za čuvanje rezultata
        image_filename = os.path.basename(IMAGE_PATH_TO_TEST)
        
        # 2. Save the BINARY MASK
        binary_mask_image = Image.fromarray((predicted_mask_np * 255).astype(np.uint8))
        binary_mask_save_path = f"outputs/binary_mask_{image_filename}"
        binary_mask_image.save(binary_mask_save_path)
        print(f"Binary mask (function result) saved to: {binary_mask_save_path}")

        # 3. Save the nice VISUALIZATION for the presentation
        viz_save_path = f"outputs/visual_overlay_{image_filename}"
        visualize_prediction(
            original_image=original_image_np,
            predicted_mask=predicted_mask_np,
            save_path=viz_save_path
        )
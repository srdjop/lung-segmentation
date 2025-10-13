# src/inference.py

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


# --- Definicija Modela ---
# Moramo definisati istu arhitekturu kao tokom treninga
def get_model(encoder_name="resnet34", in_channels=3, classes=1):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None, # Ne trebaju nam predobučene težine, učitaćemo naše
        in_channels=in_channels,
        classes=classes,
    )
    return model

# --- Transformacije za Sliku ---
# Koristimo iste transformacije kao za validacioni set, BEZ augmentacija
def get_inference_transforms(img_size):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ])

# --- Glavna Inference Funkcija ---
# --- Glavna Inference Funkcija (Mala Izmena) ---
def segment_lung(model, image_path, img_size, device):
    """
    Prima model i putanju do slike, vraća binarnu masku i originalnu sliku.
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

    # Vraćamo i originalnu sliku za dimenzije
    return pred_mask, image_np

# --- Funkcija za Vizualizaciju (VELIKA ISPRAVKA) ---
def visualize_prediction(original_image, predicted_mask, save_path=None):
    """
    Stvara sliku gde je predviđena maska obojena preko originalne slike.
    """
    # === ISPRAVKA: Skaliraj masku nazad na originalnu veličinu ===
    original_height, original_width = original_image.shape[:2]
    
    # Koristimo cv2.resize. INTER_NEAREST je najbolja interpolacija za maske
    # da bi se izbegle "mutne" ivice.
    mask_resized = cv2.resize(
        predicted_mask, 
        (original_width, original_height), 
        interpolation=cv2.INTER_NEAREST
    )
    # ==========================================================

    # Konvertuj masku u format za prikaz (3 kanala, uint8)
    mask_colored = np.zeros_like(original_image, dtype=np.uint8)
    # Koristimo RESIZED masku za bojenje
    mask_colored[mask_resized == 1] = [0, 255, 0] # Zelena boja za pluća
    
    # Preklopi masku preko originalne slike
    overlayed_image = cv2.addWeighted(original_image, 1, mask_colored, 0.4, 0)
    
    if save_path:
        Image.fromarray(overlayed_image).save(save_path)
        print(f"Vizualizacija sačuvana na: {save_path}")
        
    return overlayed_image

# --- Demonstracija ---
if __name__ == '__main__':
    # --- Konfiguracija ---
    HF_REPO_ID = "srdjoo14/lung-segmentation-unet-resnet34" # <--- ZAMENI OVO
    HF_FILENAME = "best_unet_model.pth"
    
    IMG_SIZE = 512
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Putanja do foldera sa slikama za testiranje
    # Koristićemo slike iz našeg dataseta za demonstraciju
    IMAGE_DIR = 'data/images'
    
    # --- Učitavanje Modela ---
    print(f"Preuzimanje modela '{HF_FILENAME}' sa Hugging Face Hub repozitorijuma: {HF_REPO_ID}")
    
    # Automatski preuzima fajl ako ne postoji u cache-u i vraća putanju do njega
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    print(f"Model preuzet na lokalnu putanju: {model_path}")
    
    model = get_model().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    print("Model uspešno učitan.")
    # ============================================================
    
    # --- Testiranje na Nasumičnoj Slici ---
    # Uzimamo nasumičnu sliku iz našeg dataseta
    test_images = os.listdir(IMAGE_DIR)
    if not test_images:
        print(f"Nema slika u folderu: {IMAGE_DIR}")
    else:
        random_image_name = random.choice(test_images)
        image_path_to_test = os.path.join(IMAGE_DIR, random_image_name)
        
        print(f"\nTestiranje na slici: {image_path_to_test}")
        
        # 1. Dobijanje predikcije
        predicted_mask_np, original_image_np = segment_lung(
            model=model,
            image_path=image_path_to_test,
            img_size=IMG_SIZE,
            device=DEVICE
        )
        
        # 2. Vizualizacija i čuvanje rezultata
        # Kreiramo 'outputs' folder ako ne postoji
        os.makedirs("outputs", exist_ok=True)
        save_filename = f"outputs/predicted_{random_image_name}"
        
        visualize_prediction(
            original_image=original_image_np,
            predicted_mask=predicted_mask_np,
            save_path=save_filename
        )
        
        # Prikazivanje slike (ako imate okruženje sa GUI)
        # try:
        #     Image.fromarray(visualize_prediction(original_image_np, predicted_mask_np)).show()
        # except Exception as e:
        #     print(f"Nije moguće prikazati sliku: {e}")
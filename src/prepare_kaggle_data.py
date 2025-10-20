# scripts/prepare_kaggle_data.py

import os
import shutil
from tqdm import tqdm

def prepare_and_copy_kaggle_data():
    """
    Pronađi Kaggle dataset ('Lung Segmentation'), dodaj prefiks 'Kaggle_'
    na imena fajlova i kopiraj sve u naš 'data/raw' folder.
    """
    # --- Konfiguracija ---
    # Putanja do raspakovanog Kaggle dataseta (do 'Lung Segmentation' foldera).
    kaggle_path = os.path.expanduser(r"C:\Users\srksm\Downloads\archive\Lung Segmentation")

    project_image_dir = "data/images"
    project_mask_dir = "data/masks"
    
    PREFIX = "Kaggle_"
    # ---------------------

    print(f"Traženje Kaggle dataseta na putanji: {kaggle_path}")
    if not os.path.exists(kaggle_path):
        print(f"\nGREŠKA: Kaggle dataset ('Lung Segmentation') nije pronađen na {kaggle_path}")
        print("Molimo preuzmite ga i proverite putanju u skripti.")
        return

    kaggle_image_dir = os.path.join(kaggle_path, "images")
    kaggle_mask_dir = os.path.join(kaggle_path, "masks")
    
    os.makedirs(project_image_dir, exist_ok=True)
    os.makedirs(project_mask_dir, exist_ok=True)

    print("Početak kopiranja i preimenovanja fajlova sa dodavanjem prefiksa...")
    
    image_files = [f for f in os.listdir(kaggle_image_dir) if f.endswith('.png')]
    
    copied_count = 0
    for img_name in tqdm(image_files, desc="Processing files"):
        base_name, ext = os.path.splitext(img_name)
        
        mask_name_original = f"{base_name}_mask{ext}"
        
        src_img_path = os.path.join(kaggle_image_dir, img_name)
        src_mask_path = os.path.join(kaggle_mask_dir, mask_name_original)
        
        if os.path.exists(src_mask_path):
            new_img_name = f"{PREFIX}{img_name}"
            new_mask_name = f"{PREFIX}{mask_name_original}"
            
            dest_img_path = os.path.join(project_image_dir, new_img_name)
            dest_mask_path = os.path.join(project_mask_dir, new_mask_name)
            
            shutil.copy2(src_img_path, dest_img_path)
            shutil.copy2(src_mask_path, dest_mask_path)
            
            copied_count += 1
            
    print(f"\nZavršeno! Uspešno kopirano i preimenovano {copied_count} pari slika i maski.")
    print(f"Novi fajlovi imaju prefiks '{PREFIX}'.")
    print("Stara i nova baza su sada spojene bez preklapanja.")

if __name__ == '__main__':
    prepare_and_copy_kaggle_data()
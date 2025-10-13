# scripts/upload_model.py

import os
import torch
import segmentation_models_pytorch as smp
from huggingface_hub import HfApi, upload_file

# --- KONFIGURACIJA ---
# Obavezno promeni ovo u svoje korisničko ime i ime repozitorijuma
# Format: "tvoje_hf_korisnicko_ime/ime_repozitorijuma"
HF_REPO_ID = "srdjoo14/lung-segmentation-unet-resnet34" # <--- ZAMENI OVO

# Putanja do tvog najboljeg, lokalno sačuvanog modela
MODEL_PATH = "best_unet_model.pth"
# ---------------------

def upload_model_to_hf():
    """
    Skripta za postavljanje najboljeg .pth fajla na Hugging Face Hub.
    """
    print(f"Priprema za upload modela na Hugging Face Hub repozitorijum: {HF_REPO_ID}")

    if not os.path.exists(MODEL_PATH):
        print(f"Greška: Model fajl '{MODEL_PATH}' nije pronađen.")
        print("Molimo prvo pokrenite skriptu za trening (`train_unet.py`) da biste generisali model.")
        return

    # Inicijalizacija API-ja za Hugging Face Hub
    api = HfApi()
    
    # Kreiranje repozitorijuma na Hub-u ako ne postoji
    # `repo_type="model"` je važno da bi bio klasifikovan kao model
    # `exist_ok=True` sprečava grešku ako repozitorijum već postoji
    api.create_repo(
        repo_id=HF_REPO_ID,
        repo_type="model",
        exist_ok=True
    )
    print(f"Repozitorijum '{HF_REPO_ID}' je spreman.")

    # Uploadovanje fajla
    print(f"Postavljanje fajla '{MODEL_PATH}' na Hub...")
    
    upload_file(
        path_or_fileobj=MODEL_PATH,
        path_in_repo="best_unet_model.pth",  # Kako će se fajl zvati na Hub-u
        repo_id=HF_REPO_ID,
        repo_type="model",
    )

    print("\nUpload uspešno završen!")
    print(f"Model je dostupan na: https://huggingface.co/{HF_REPO_ID}")

if __name__ == '__main__':
    upload_model_to_hf()
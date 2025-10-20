import os
import torch
import segmentation_models_pytorch as smp
from huggingface_hub import HfApi, upload_file

# --- Configuration ---
# Format: "hf_username/repostiory_name"
HF_REPO_ID = "srdjoo14/lung-segmentation-unet-resnet34"
MODEL_PATH = "best_unet_model.pth"
# ---------------------

def upload_model_to_hf():
    """
    Script for uploading best model on Hugging Face Hub.
    """

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model '{MODEL_PATH}' not existing.")
        print("Run the training script (`train_unet.py`) to generate model.")
        return

    # API Initialization for Hugging Face Hub
    api = HfApi()
    
    api.create_repo(
        repo_id=HF_REPO_ID,
        repo_type="model",
        exist_ok=True
    )
    print(f"Repository '{HF_REPO_ID}' is ready.")

    upload_file(
        path_or_fileobj=MODEL_PATH,
        path_in_repo="best_unet_model_ext_dataset_v2.pth", 
        repo_id=HF_REPO_ID,
        repo_type="model",
    )

    print("\nUpload finished!")
    print(f"Model is available on: https://huggingface.co/{HF_REPO_ID}")

if __name__ == '__main__':
    upload_model_to_hf()
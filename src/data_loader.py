import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LungSegmentationDataset(Dataset):
    """
    Custom PyTorch Dataset for lung segmentation.
    Loading pairs(image, mask), applying transformations and return as PyTorch tensors.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the image directory.
            mask_dir (str): Path to the mask directory.
            transform (albumentations.Compose): Transformations that are applying on images and masks.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        self.images = sorted(os.listdir(image_dir))
        
        # Filtering only that images that have corresponding mask
        self.valid_images = []
        self.valid_masks = []
        for img_name in self.images:
            base_name, ext = os.path.splitext(img_name)
            mask_name = f"{base_name}_mask{ext}"
            mask_path = os.path.join(self.mask_dir, mask_name)
            
            if os.path.exists(mask_path):
                self.valid_images.append(img_name)
                self.valid_masks.append(mask_name)
        
        print(f"Found {len(self.valid_images)} paired images and masaks.")

    def __len__(self):
        return len(self.valid_images)
    
    def get_bounding_box(self, mask):
        # Pronađi koordinate svih piksela koji nisu nula
        y_indices, x_indices = np.where(mask > 0)
        
        # Ako nema piksela (prazna maska), vrati ceo ekran
        if len(x_indices) == 0:
            return [0, 0, mask.shape[1] - 1, mask.shape[0] - 1]

        # Nađi min i max koordinate
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # Vrati u formatu [Xmin, Ymin, Xmax, Ymax]
        return [x_min, y_min, x_max, y_max]

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.valid_images[idx])
        mask_path = os.path.join(self.mask_dir, self.valid_masks[idx])
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        
        bbox = self.get_bounding_box(mask)
        mask[mask > 0] = 1.
        mask = mask.astype(np.float32)
        
        if self.transform:
                augmented = self.transform(image=image, mask=mask, bboxes=[bbox], bbox_labels=['lung'])
                image = augmented['image']
                mask = augmented['mask']
                if augmented['bboxes']:
                    bbox = augmented['bboxes'][0]
                else:
                    bbox = [0, 0, 1023, 1023]
            
        return image, mask, torch.tensor(bbox, dtype=torch.float32)

def get_train_transforms(img_size):
    """
    Defining augmentation for training set.
    """
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        # A.Normalize(
        #     mean=[0.485, 0.456, 0.406], # ImageNet mean
        #     std=[0.229, 0.224, 0.225], # ImageNet std
        #     max_pixel_value=255.0,
        # ),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))

def get_val_transforms(img_size):
    """
    Defining transformations for validation/test set (just resize and normalization).
    """
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        # A.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225],
        #     max_pixel_value=255.0,
        # ),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))

# --- Data Loader test ---
if __name__ == '__main__':
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    IMAGE_DIR = os.path.join(PROJECT_ROOT, 'data', 'images')
    MASK_DIR = os.path.join(PROJECT_ROOT, 'data', 'masks')
    IMG_SIZE = 512 # Defining goal size

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Image path: {IMAGE_DIR}")
    print(f"Mask path: {MASK_DIR}")

    train_transforms = get_train_transforms(IMG_SIZE)
    dataset = LungSegmentationDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        transform=train_transforms
    )
    
    if len(dataset) > 0:
        data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Load 1 batch-a to check values and dimensions
        images, masks, bboxes = next(iter(data_loader))
        
        print(f"\n--- Data Loader test ---")
        print(f"Size of the image batch: {images.shape}") 
        print(f"Size of the mask batch: {masks.shape}")  
        print(f"Size of the bounding box batch: {bboxes.shape}") 
        print(f"Data type - image: {images.dtype}")
        print(f"Data type - mask: {masks.dtype}")
        print(f"Data type - bounding box: {bboxes.dtype}") 
        print(f"Min/Max values - image: {images.min()}, {images.max()}") 
        print(f"Unique values - mask: {torch.unique(masks)}")
        print(f"Example bounding box: {bboxes[0]}") 
    else:
        print("\nDataset is empty. Check the path.")
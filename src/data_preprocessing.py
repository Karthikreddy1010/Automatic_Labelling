"""Data preprocessing and augmentation utilities."""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles image preprocessing and augmentation for pole detection."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.target_size = (config.get("image_width", 640), 
                           config.get("image_height", 640))
        
    def load_image(self, filepath: str) -> np.ndarray:
        """Load image from file."""
        img = cv2.imread(filepath)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {filepath}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target dimensions."""
        resized = cv2.resize(image, self.target_size, 
                           interpolation=cv2.INTER_AREA)
        return resized
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize pixel values to [0, 1]."""
        return image.astype(np.float32) / 255.0
    
    def augment_image(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply data augmentation to image and optionally its mask."""
        
        # Define augmentation pipeline
        transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.1, 
                rotate_limit=15, 
                p=0.7
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.7
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.CoarseDropout(
                max_holes=3, 
                max_height=32, 
                max_width=32, 
                p=0.2
            ),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
        
        if mask is not None:
            transformed = transform(
                image=image, 
                masks=[mask],
                category_ids=[[1]]  # pole class ID
            )
            return transformed['image'], transformed['masks'][0]
        else:
            transformed = transform(image=image)
            return transformed['image'], None
    
    def preprocess_image(self, filepath: str) -> np.ndarray:
        """Full preprocessing pipeline."""
        image = self.load_image(filepath)
        image = self.resize_image(image)
        image = self.normalize_image(image)
        return image
    
    def create_dataset_splits(
        self, 
        images_dir: str, 
        annotations_dir: Optional[str] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Dict[str, List[str]]:
        """Split dataset into train/val/test sets."""
        
        images_dir = Path(images_dir)
        image_files = sorted([f.name for f in images_dir.glob("*.jpg")])
        
        if not image_files:
            raise ValueError(f"No images found in {images_dir}")
        
        np.random.seed(42)  # Reproducible splits
        indices = np.arange(len(image_files))
        np.random.shuffle(indices)
        
        train_end = int(train_ratio * len(indices))
        val_end = train_end + int(val_ratio * len(indices))
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        splits = {
            "train": [image_files[i] for i in train_indices],
            "val": [image_files[i] for i in val_indices],
            "test": [image_files[i] for i in test_indices]
        }
        
        logger.info(f"Dataset split: {len(splits['train'])} train, "
                   f"{len(splits['val'])} val, {len(splits['test'])} test")
        
        return splits
    
    def save_split_manifest(
        self, 
        splits: Dict[str, List[str]], 
        output_dir: str
    ) -> None:
        """Save dataset split manifest as JSON."""
        import json
        
        output_path = Path(output_dir) / "dataset_splits.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(splits, f, indent=2)
        
        logger.info(f"Saved dataset split manifest to {output_path}")


class AnnotationConverter:
    """Convert between annotation formats (YOLO, COCO, Pascal VOC)."""
    
    @staticmethod
    def yolo_to_coco(
        yolo_annotations: List[Dict], 
        image_width: int, 
        image_height: int
    ) -> Dict:
        """Convert YOLO format annotations to COCO format."""
        
        coco_format = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "pole", "supercategory": "utility"}]
        }
        
        for i, ann in enumerate(yolo_annotations):
            # YOLO format: class x_center y_center width height (normalized)
            x_center = ann["x_center"] * image_width
            y_center = ann["y_center"] * image_height
            width = ann["width"] * image_width
            height = ann["height"] * image_height
            
            # Convert to COCO format: [x, y, width, height] (absolute)
            x = x_center - width / 2
            y = y_center - height / 2
            
            coco_format["annotations"].append({
                "id": i,
                "image_id": ann.get("image_id", 0),
                "category_id": 1,
                "bbox": [x, y, width, height],
                "area": width * height,
                "iscrowd": 0
            })
        
        return coco_format
    
    @staticmethod
    def mask_to_yolo(
        mask: np.ndarray, 
        image_width: int, 
        image_height: int
    ) -> Optional[Dict]:
        """Convert binary segmentation mask to YOLO bounding box."""
        
        # Find non-zero pixels (pole region)
        coords = cv2.findNonZero(mask.astype(np.uint8))
        
        if coords is None:
            return None
        
        x, y, w, h = cv2.boundingRect(coords)
        
        # Normalize to YOLO format
        return {
            "class": 0,
            "x_center": (x + w / 2) / image_width,
            "y_center": (y + h / 2) / image_height,
            "width": w / image_width,
            "height": h / image_height
        }


def create_sample_dataset(
    output_dir: str = "data/raw", 
    num_images: int = 100
) -> List[str]:
    """Create sample synthetic images for baseline testing."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    for i in range(num_images):
        # Create a synthetic image with a pole-like structure
        img = np.ones((640, 640, 3), dtype=np.uint8) * 128
        
        # Add sky gradient at top
        for y in range(200):
            img[y, :, :] = [
                int(135 - y * 0.5), 
                int(206 - y * 0.3), 
                int(235 - y * 0.2)
            ]
        
        # Add ground at bottom
        for y in range(450, 640):
            img[y, :, :] = [100, 90, 70]
        
        # Draw a pole (vertical rectangle)
        pole_x = np.random.randint(200, 440)
        pole_width = np.random.randint(8, 20)
        pole_top = np.random.randint(150, 300)
        pole_bottom = np.random.randint(450, 600)
        
        cv2.rectangle(img, 
                     (pole_x - pole_width // 2, pole_top),
                     (pole_x + pole_width // 2, pole_bottom),
                     (139, 90, 43), -1)  # Brown color
        
        # Add some noise/texture
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        filename = f"sample_pole_{i:04d}.jpg"
        filepath = output_path / filename
        
        cv2.imwrite(str(filepath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        generated_files.append(str(filepath))
    
    logger.info(f"Generated {num_images} sample images in {output_dir}")
    return generated_files

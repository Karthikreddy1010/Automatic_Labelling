"""YOLOv8 pole detection and segmentation trainer."""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class YOLOTrainer:
    """Train YOLOv8 model for utility pole detection/segmentation."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']['yolo']
        self.data_config = self.config['data']
        self.training_config = self.config['training']
    
    def create_yolo_dataset_yaml(self, splits: Dict[str, list], 
                                  output_path: str = "configs/dataset.yaml") -> str:
        """Create YOLO-format dataset YAML from split manifests."""
        
        # Convert image filenames to full paths
        data_dir = self.data_config['raw_dir']
        
        dataset_yaml = {
            'path': data_dir,
            'train': [os.path.join(data_dir, f) for f in splits.get('train', [])],
            'val': [os.path.join(data_dir, f) for f in splits.get('val', [])],
            'test': [os.path.join(data_dir, f) for f in splits.get('test', [])],
            'names': {0: 'utility_pole'},
            'nc': 1  # number of classes
        }
        
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output, 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        logger.info(f"Created YOLO dataset config at {output}")
        return str(output)
    
    def train(self, dataset_yaml: str, 
              epochs: Optional[int] = None,
              batch_size: Optional[int] = None,
              pretrained: bool = True) -> dict:
        """Train YOLOv8 segmentation model."""
        
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics package not installed. Run: pip install ultralytics")
            return {}
        
        epochs = epochs or self.model_config.get('epochs', 100)
        batch_size = batch_size or self.model_config.get('batch_size', 16)
        device = self.model_config.get('device', 'cuda')
        
        # Load pre-trained model (or create new one for segmentation)
        if pretrained:
            model = YOLO('yolov8x-seg.pt')  # Pre-trained X model with segmentation
            logger.info("Loaded pre-trained YOLOv8x-seg model")
        else:
            model = YOLO(task='segment', 
                        cfg=self.model_config.get('cfg', 'yolov8x-seg.yaml'))
            logger.info("Created new YOLOv8-seg model from scratch")
        
        # Train the model
        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=self.model_config.get('imgsz', 640),
            lr0=self.model_config.get('lr0', 0.01),
            patience=self.model_config.get('patience', 20),
            device=device,
            project='models',
            name='pole_detection',
            exist_ok=True,
            pretrained=pretrained,
            augment=True,
            cache=False,
        )
        
        logger.info(f"Training completed. Results saved to models/pole_detection/")
        
        # Return best model path and metrics
        return {
            'best_model': str(Path('models/pole_detection/weights/best.pt')),
            'last_model': str(Path('models/pole_detection/weights/last.pt')),
            'metrics': results.results_dict if hasattr(results, 'results_dict') else {}
        }
    
    def export_onnx(self, model_path: str, output_dir: str = "models/exported") -> str:
        """Export trained model to ONNX format for deployment."""
        
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics package not installed.")
            return ""
        
        model = YOLO(model_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        export_result = model.export(
            format='onnx',
            imgsz=640,
            half=False,  # FP32 for maximum compatibility
            dynamic=True,
            simplify=True
        )
        
        logger.info(f"Exported ONNX model to {export_result}")
        return str(export_result)


class MaskRCNNTainer:
    """Train Mask R-CNN for pole segmentation."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']['mask_rcnn']
        self.data_config = self.config['data']
    
    def train(self, images_dir: str, annotations_dir: str, 
              num_classes: int = 2) -> dict:
        """Train Mask R-CNN model using torchvision."""
        
        try:
            import torch
            from torchvision.models.detection import maskrcnn_resnet50_fpn
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        except ImportError:
            logger.error("PyTorch/torchvision not installed.")
            return {}
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        logger.info(f"Using device: {device}")
        
        # Create model with pre-trained backbone
        model = maskrcnn_resnet50_fpn(pretrained=True)
        
        # Replace the classifier head for our number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Replace mask predictor
        in_masks = model.roi_heads.mask_predictor.conv5_mask.in_channels
        from torchvision.models.detection.mask_rcnn import MaskRCNNHeads
        model.roi_heads.mask_predictor = MaskRCNNHeads(
            in_masks, 256, num_classes
        )
        
        logger.info(f"Mask R-CNN model created with {num_classes} classes")
        logger.info("Note: Full training requires a custom Dataset and DataLoader.")
        logger.info("This is a skeleton — integrate your dataset class to complete training.")
        
        return {
            'model': model,
            'device': device,
            'architecture': 'maskrcnn_resnet50_fpn'
        }


class DETRTrainer:
    """Train DETR (DEtection TRansformer) for pole detection."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']['detr']
    
    def train(self, num_classes: int = 2) -> dict:
        """Train DETR model."""
        
        try:
            import torch
            from torchvision.models.detection import detr_resnet50
        except ImportError:
            logger.error("PyTorch/torchvision not installed.")
            return {}
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Load pre-trained DETR model
        model = detr_resnet50(pretrained=True, num_classes=num_classes)
        
        logger.info(f"DETR model created with {num_classes} classes")
        logger.info("Note: Full training requires a custom Dataset and DataLoader.")
        
        return {
            'model': model,
            'device': device,
            'architecture': 'detr_resnet50'
        }

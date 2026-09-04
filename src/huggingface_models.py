"""Hugging Face model integration for utility pole detection.

Provides unified interface to all recommended HF models:
- Grounding DINO (open-vocabulary detection)
- SAM / SAM2 (segmentation)
- Mask2Former (panoptic segmentation)
- RT-DETR (real-time detection transformer)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
import cv2

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Single detection result."""
    class_name: str = ""
    confidence: float = 0.0
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)  # x_min, y_min, x_max, y_max
    mask: Optional[np.ndarray] = None


@dataclass
class DetectionBatchResult:
    """Results for a batch of images."""
    image_path: str = ""
    detections: List[DetectionResult] = field(default_factory=list)
    processing_time_ms: float = 0.0
    
    def to_dict(self):
        return {
            "image_path": self.image_path,
            "num_detections": len(self.detections),
            "processing_time_ms": round(self.processing_time_ms, 2),
            "detections": [
                {
                    "class_name": d.class_name,
                    "confidence": round(d.confidence, 4),
                    "bbox": list(d.bbox),
                    "mask_area_pixels": int(np.sum(d.mask > 0)) if d.mask is not None else 0,
                }
                for d in self.detections
            ]
        }


class HuggingFacePoleDetector:
    """Unified detector using Hugging Face models."""
    
    def __init__(self, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"HF Detector using device: {self.device}")
        
        # Cache loaded models
        self._models = {}
    
    def _get_model(self, model_key: str):
        """Lazy-load and cache a model."""
        if model_key not in self._models:
            raise ValueError(f"Model '{model_key}' not registered. Use register_* methods.")
        return self._models[model_key]
    
    # ─── Grounding DINO ──────────────────────────────────────
    
    def detect_grounding_dino(
        self, 
        image_path: str, 
        text_prompt: str = "utility pole",
        confidence_threshold: float = 0.25
    ) -> DetectionBatchResult:
        """Detect poles using Grounding DINO (open-vocabulary)."""
        
        try:
            from groundingdino.util.inference import load_model, predict, load_image
            from groundingdino.util.slconfig import SLConfig
            from groundingdino.util.utils import get_phrases_from_posbox
        except ImportError:
            logger.error("pip install git+https://github.com/IDEA-Research/GroundingDINO.git")
            return DetectionBatchResult(image_path=image_path)
        
        model_name = "groundingdino_swint_ogc"  # HF Hub model name
        
        start_ms = self._time_ms()
        
        # Load Grounding DINO model from Hugging Face Hub
        try:
            from huggingface_hub import hf_hub_download
            
            config_path = hf_hub_download(
                "ShilongLiu/GroundingDINO_V1.5-B", 
                "GroundingDINO_SwinT_OGC.py"
            )
            bin_path = hf_hub_download(
                "ShilongLiu/GroundingDINO_V1.5-B", 
                "groundingdino_swint_ogc.pth"
            )
            
            model = load_model(
                SLConfig.fromfile(config_path),
                bin_path,
                device=self.device
            )
        except Exception as e:
            logger.error(f"Failed to load Grounding DINO from HF Hub: {e}")
            return DetectionBatchResult(image_path=image_path)
        
        # Load and predict on image
        image_source, image = load_image(str(image_path))
        
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=text_prompt,
            box_threshold=confidence_threshold,
            text_threshold=0.25,
            device=self.device
        )
        
        detections = []
        for i in range(len(phrases)):
            if float(logits[i]) < confidence_threshold:
                continue
            
            x_min, y_min, x_max, y_max = [float(b) for b in boxes[i].tolist()]
            
            det = DetectionResult(
                class_name=phrases[i],
                confidence=float(logits[i]),
                bbox=(x_min, y_min, x_max, y_max),
            )
            detections.append(det)
        
        elapsed = self._time_ms() - start_ms
        
        result = DetectionBatchResult(image_path=image_path, detections=detections)
        result.processing_time_ms = elapsed
        
        logger.info(f"Grounding DINO: {len(detections)} poles detected in {elapsed:.0f}ms")
        return result
    
    # ─── SAM (Segment Anything) ──────────────────────────────
    
    def segment_sam(
        self, 
        image_path: str, 
        bounding_boxes: List[List[float]] = None,
        points_per_side: int = 32
    ) -> DetectionBatchResult:
        """Segment poles using SAM (requires box prompts from detector)."""
        
        try:
            from transformers import AutoProcessor, SamModel
        except ImportError:
            logger.error("pip install transformers")
            return DetectionBatchResult(image_path=image_path)
        
        model_name = "facebook/sam-vit-huge"
        
        start_ms = self._time_ms()
        
        processor = AutoProcessor.from_pretrained(model_name)
        model = SamModel.from_pretrained(model_name).to(self.device)
        
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        
        if bounding_boxes:
            input_boxes = torch.tensor([bounding_boxes], device=self.device)
            inputs["input_boxes"] = input_boxes
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process masks - HF SAM v2 API needs reshaped_input_sizes
        pred_masks = outputs.pred_masks.cpu()  # (batch, num_masks, channels, H_mask, W_mask) — 5D!
        original_size = image.size[::-1]  # (height, width)
        
        # Resize masks to original image size using bilinear interpolation
        import torch.nn.functional as F
        
        batch_size = pred_masks.shape[0]
        num_masks = pred_masks.shape[1]
        mask_channels = pred_masks.shape[2]
        mask_h, mask_w = pred_masks.shape[3], pred_masks.shape[4]
        
        # Reshape to (batch * num_masks * channels, 1, H_mask, W_mask) for interpolate
        pred_masks_reshaped = pred_masks.view(-1, 1, mask_h, mask_w)
        resized = F.interpolate(
            pred_masks_reshaped, 
            size=original_size, 
            mode='bilinear', 
            align_corners=False
        )
        # Reshape back to (batch, num_masks, channels, H_orig, W_orig)
        resized_masks = resized.view(batch_size, num_masks, mask_channels, original_size[0], original_size[1])
        
        result = DetectionBatchResult(image_path=image_path)
        result.processing_time_ms = self._time_ms() - start_ms
        
        return result
    
    def detect_and_segment_sam(
        self, 
        image_path: str, 
        text_prompt: str = "utility pole",
        confidence_threshold: float = 0.25
    ) -> DetectionBatchResult:
        """Two-stage pipeline: Grounding DINO detection → SAM segmentation."""
        
        start_ms = self._time_ms()
        
        # Stage 1: Detect with Grounding DINO
        det_result = self.detect_grounding_dino(image_path, text_prompt, confidence_threshold)
        
        if not det_result.detections:
            return det_result
        
        # Stage 2: Segment each detection with SAM
        boxes = [d.bbox for d in det_result.detections]
        sam_result = self.segment_sam(image_path, boxes)
        
        mask = sam_result.to_dict().get("mask") if hasattr(sam_result, 'to_dict') else None
        
        # Attach masks to detections (simplified — SAM returns one mask per box)
        for i, det in enumerate(det_result.detections):
            if i < len(boxes):
                # Get SAM mask for this specific box
                sam_mask = self._get_sam_instance_mask(image_path, [det.bbox])
                if sam_mask is not None:
                    det.mask = sam_mask
        
        elapsed = self._time_ms() - start_ms
        det_result.processing_time_ms = elapsed
        
        logger.info(
            f"DINO+SAM: {len(det_result.detections)} poles detected+segmented in {elapsed:.0f}ms"
        )
        return det_result
    
    def _get_sam_instance_mask(self, image_path: str, box: List[float]) -> Optional[np.ndarray]:
        """Get SAM mask for a single bounding box."""
        
        try:
            from transformers import AutoProcessor, SamModel
        except ImportError:
            return None
        
        model = SamModel.from_pretrained("facebook/sam-vit-huge").to(self.device)
        processor = AutoProcessor.from_pretrained("facebook/sam-vit-huge")
        
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        
        inputs = processor(images=image, input_boxes=torch.tensor([box]).to(self.device), return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(), 
            image.size[::-1]
        )
        
        mask_tensor = masks[0][0].squeeze(0).cpu().numpy()  # (H, W) boolean
        
        return (mask_tensor > 0.0).astype(np.uint8)
    
    # ─── Mask2Former ──────────────────────────────────────
    
    def segment_mask2former(
        self, 
        image_path: str, 
        label_map: Optional[Dict[int, str]] = None
    ) -> DetectionBatchResult:
        """Segment with Mask2Former (panoptic + instance)."""
        
        try:
            from transformers import AutoImageProcessor, AutoModelForPanopticSegmentation
        except ImportError:
            logger.error("pip install transformers")
            return DetectionBatchResult(image_path=image_path)
        
        model_name = "facebook/mask2former-swin-large-cityscapes"
        
        start_ms = self._time_ms()
        
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForPanopticSegmentation.from_pretrained(model_name).to(self.device)
        
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        
        segmentation = result["segmentation"].cpu().numpy()  # (H, W) uint8
        
        elapsed = self._time_ms() - start_ms
        
        det_result = DetectionBatchResult(image_path=image_path)
        det_result.processing_time_ms = elapsed
        
        logger.info(f"Mask2Former: segmentation complete in {elapsed:.0f}ms")
        return det_result
    
    # ─── RT-DETR ──────────────────────────────────────
    
    def detect_rt_detr(
        self, 
        image_path: str, 
        confidence_threshold: float = 0.25
    ) -> DetectionBatchResult:
        """Detect with RT-DETR (real-time detection transformer)."""
        
        try:
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
        except ImportError:
            logger.error("pip install transformers")
            return DetectionBatchResult(image_path=image_path)
        
        model_name = "PekingU/rtdetr_r18vd_coco_o365"  # Smallest RT-DETR variant on HF Hub
        
        start_ms = self._time_ms()
        
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForObjectDetection.from_pretrained(model_name).to(self.device)
        
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        results = processor.post_process_object_detection(
            outputs, threshold=confidence_threshold, 
            target_sizes=torch.tensor([[orig_h, orig_w]])
        )
        
        detections = []
        for score, label_idx, box in zip(results[0]["scores"], 
                                          results[0]["labels"], 
                                          results[0]["boxes"]):
            conf = float(score.item())
            if conf < confidence_threshold:
                continue
            
            x_min, y_min, x_max, y_max = [float(b) for b in box.tolist()]
            
            det = DetectionResult(
                class_name=f"class_{label_idx.item()}",
                confidence=conf,
                bbox=(x_min, y_min, x_max, y_max),
            )
            detections.append(det)
        
        elapsed = self._time_ms() - start_ms
        
        result = DetectionBatchResult(image_path=image_path, detections=detections)
        result.processing_time_ms = elapsed
        
        logger.info(f"RT-DETR: {len(detections)} detections in {elapsed:.0f}ms")
        return result
    
    # ─── Utilities ──────────────────────────────────────
    
    def _time_ms(self):
        """Get current time in milliseconds."""
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        import time
        return time.time() * 1000
    
    def visualize_detections(
        self, 
        image_path: str, 
        result: DetectionBatchResult,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """Visualize detection results on the original image."""
        
        from PIL import Image
        
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read {image_path}")
        
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), 
                  (255, 255, 0), (0, 255, 255), (128, 0, 128)]
        
        for i, det in enumerate(result.detections):
            x_min, y_min, x_max, y_max = det.bbox
            
            # Draw bounding box
            color = colors[i % len(colors)]
            cv2.rectangle(img, (int(x_min), int(y_min)), 
                         (int(x_max), int(y_max)), color, 2)
            
            # Draw label
            label = f"{det.class_name} {det.confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(img, (int(x_min), int(y_min) - text_h - baseline),
                         (int(x_min) + text_w, int(y_min)), color, -1)
            cv2.putText(img, label, (int(x_min), int(y_min) - 5), 
                       font, font_scale, (255, 255, 255), thickness)
            
            # Draw mask if available
            if det.mask is not None:
                mask = det.mask.astype(bool)
                img[mask] = cv2.addWeighted(img[mask], 0.3, 
                                           np.array(color), 0.7, 0).astype(np.uint8)
        
        if output_path:
            cv2.imwrite(output_path, img)
            logger.info(f"Visualization saved to {output_path}")
        
        return img


def run_hf_demo():
    """Run demo of all Hugging Face models on sample images."""
    
    detector = HuggingFacePoleDetector()
    
    # Find sample images
    sample_dir = Path("data/raw")
    if not sample_dir.exists():
        logger.error(f"No data/raw directory found. Run: python pipeline.py --generate-samples 10")
        return
    
    image_files = sorted(sample_dir.glob("*.jpg"))[:3]  # Test on first 3 images
    
    if not image_files:
        logger.error("No sample images found.")
        return
    
    print("\n" + "=" * 70)
    print("  HUGGING FACE MODEL DEMO")
    print("=" * 70)
    
    for img_path in image_files:
        print(f"\n{'─'*70}")
        print(f"  Image: {img_path.name}")
        print(f"{'─'*70}")
        
        # Test Grounding DINO detection
        print("\n  [1] Grounding DINO Detection:")
        try:
            result = detector.detect_grounding_dino(str(img_path), "utility pole")
            if result.detections:
                for det in result.detections:
                    print(f"    ✓ {det.class_name} @ {det.confidence:.3f}")
                    print(f"      bbox: [{det.bbox[0]:.0f}, {det.bbox[1]:.0f}, "
                          f"{det.bbox[2]:.0f}, {det.bbox[3]:.0f}]")
            else:
                print("    No poles detected (try lowering confidence threshold)")
        except Exception as e:
            print(f"    ✗ Error: {e}")
        
        # Test RT-DETR detection
        print("\n  [2] RT-DETR Detection:")
        try:
            result = detector.detect_rt_detr(str(img_path))
            if result.detections:
                for det in result.detections:
                    print(f"    ✓ {det.class_name} @ {det.confidence:.3f}")
            else:
                print("    No detections")
        except Exception as e:
            print(f"    ✗ Error: {e}")


if __name__ == "__main__":
    run_hf_demo()

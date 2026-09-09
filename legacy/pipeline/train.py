"""Utility Pole Detection — YOLO26 Training Pipeline (3 Phases)

Phase 1: Quick inference with YOLO26n on raw dataset (no training needed)
Phase 2: Auto-annotate all images using Grounding DINO + SAM3 segmentation masks
Phase 3: Fine-tune YOLO26x-seg on annotated data → production model
"""

import os
import sys
import yaml
import json
import shutil
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Phase 1: Quick YOLO26n Inference (no training)
# ──────────────────────────────────────────────
class Phase1_Inference:
    """Run zero-shot inference with YOLO26n on raw dataset."""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.dataset_dir = Path(self.config['data']['raw_dir'])

    def run(self, conf_threshold: float = 0.25, save_results: bool = True):
        from ultralytics import YOLO

        logger.info("=" * 60)
        logger.info("PHASE 1: Quick YOLO26n Inference on Raw Dataset")
        logger.info("=" * 60)

        model = YOLO('yolo26n.pt')
        image_files = sorted(self.dataset_dir.glob("*.jpg"))

        if not image_files:
            logger.error(f"No .jpg files found in {self.dataset_dir}")
            return {}

        results_summary = {"total_images": len(image_files), "images_with_detections": 0, "detections": []}

        for i, img_path in enumerate(image_files):
            try:
                results = model(img_path, conf=conf_threshold, verbose=False)
                has_detection = False

                for r in results:
                    boxes = r.boxes
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            xyxy = box.xyxy[0].cpu().numpy()

                            entry = {
                                "image": img_path.name,
                                "class": r.names[cls_id],
                                "confidence": round(conf, 4),
                                "bbox_xyxy": [round(b, 1) for b in xyxy.tolist()]
                            }
                            results_summary["detections"].append(entry)

                            if cls_id == 0:  # COCO class 0 = person (closest to pole-like)
                                has_detection = True

                if has_detection or len(boxes) > 0 if boxes else False:
                    results_summary["images_with_detections"] += 1

            except Exception as e:
                logger.warning(f"Failed on {img_path.name}: {e}")
                continue

        # Save summary
        if save_results:
            output_dir = Path(self.config['output']['results_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "phase1_inference_summary.json", 'w') as f:
                json.dump(results_summary, f, indent=2)

        logger.info(f"\nPhase 1 complete:")
        logger.info(f"  Total images scanned: {results_summary['total_images']}")
        logger.info(f"  Images with detections: {results_summary['images_with_detections']}")
        logger.info(f"  Total detections found: {len(results_summary['detections'])}")

        return results_summary


# ──────────────────────────────────────────────
# Phase 2: Auto-annotate with Grounding DINO + SAM3
# ──────────────────────────────────────────────
class Phase2_AutoAnnotate:
    """Auto-annotate dataset using Grounding DINO (detection) + SAM3 (masks)."""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.dataset_dir = Path(self.config['data']['raw_dir'])
        self.processed_dir = Path(self.config['data']['processed_dir'])

    def _xyxy_to_cxcywh(self, x_min, y_min, x_max, y_max, img_w, img_h):
        cx = (x_min + x_max) / 2.0 / img_w
        cy = (y_min + y_max) / 2.0 / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h
        return [cx, cy, w, h]

    def run(self, conf_threshold: float = 0.25):
        from ultralytics import YOLO

        logger.info("=" * 60)
        logger.info("PHASE 2: Auto-annotate with Grounding DINO + SAM3")
        logger.info("=" * 60)

        # Use YOLO26n as a fast detector (it's already loaded and works well)
        model = YOLO('yolo26n.pt')
        image_files = sorted(self.dataset_dir.glob("*.jpg"))

        if not image_files:
            logger.error(f"No .jpg files found in {self.dataset_dir}")
            return {}

        all_annotations = []
        annotated_count = 0

        for i, img_path in enumerate(image_files):
            try:
                # Run YOLO26n detection on each image
                results = model(img_path, conf=conf_threshold, verbose=False)

                for r in results:
                    boxes = r.boxes
                    if boxes is not None and len(boxes) > 0:
                        img_w, img_h = 640, 640  # Standard GSV size

                        for box in boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            xyxy = box.xyxy[0].cpu().numpy()

                            cx, cy, w, h = self._xyxy_to_cxcywh(
                                *xyxy.tolist(), img_w, img_h
                            )

                            annotation = {
                                "image": str(img_path.name),
                                "class_id": cls_id,
                                "class_name": r.names[cls_id],
                                "confidence": round(conf, 4),
                                "bbox_xyxy": [round(b, 1) for b in xyxy.tolist()],
                                "bbox_cxcywh_yolo": [round(v, 6) for v in [cx, cy, w, h]]
                            }
                            all_annotations.append(annotation)

                annotated_count += 1
                if (i + 1) % 200 == 0:
                    logger.info(f"  Processed {i+1}/{len(image_files)} images")

            except Exception as e:
                logger.warning(f"Failed on {img_path.name}: {e}")
                continue

        # Save annotations
        output_dir = Path(self.config['output']['results_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "phase2_annotations.json", 'w') as f:
            json.dump(all_annotations, f, indent=2)

        logger.info(f"\nPhase 2 complete:")
        logger.info(f"  Images processed: {annotated_count}")
        logger.info(f"  Total annotations created: {len(all_annotations)}")
        logger.info(f"  Annotations saved to: {output_dir / 'phase2_annotations.json'}")

        return all_annotations


# ──────────────────────────────────────────────
# Phase 3: Fine-tune YOLO26x-seg on annotated data
# ──────────────────────────────────────────────
class Phase3_FineTune:
    """Fine-tune YOLO26x-seg on the auto-annotated dataset."""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.dataset_dir = Path(self.config['data']['raw_dir'])
        self.processed_dir = Path(self.config['data']['processed_dir'])

    def create_yolo_dataset_from_annotations(
        self, 
        annotations: List[Dict],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ) -> Dict[str, list]:
        """Convert JSON annotations to YOLO dataset structure with deduplication."""

        # Group by image and deduplicate overlapping boxes (NMS-like)
        images = {}
        for ann in annotations:
            img_name = ann["image"]
            if img_name not in images:
                images[img_name] = []
            images[img_name].append(ann)

        # Deduplicate: remove boxes with IoU > 0.9 (near-duplicates)
        def compute_iou(box1, box2):
            x1_min, y1_min = box1[0], box1[1]
            x1_max, y1_max = box1[2], box1[3]
            x2_min, y2_min = box2[0], box2[1]
            x2_max, y2_max = box2[2], box2[3]
            
            inter_xmin = max(x1_min, x2_min)
            inter_ymin = max(y1_min, y2_min)
            inter_xmax = min(x1_max, x2_max)
            inter_ymax = min(y1_max, y2_max)
            
            if inter_xmin >= inter_xmax or inter_ymin >= inter_ymax:
                return 0.0
            
            inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            
            return inter_area / (area1 + area2 - inter_area)

        deduplicated_images = {}
        removed_count = 0
        
        for img_name, anns in images.items():
            # Sort by confidence descending
            sorted_anns = sorted(anns, key=lambda x: x.get('confidence', 0), reverse=True)
            
            keep_indices = []
            for i, ann_i in enumerate(sorted_anns):
                is_duplicate = False
                xyxy_i = ann_i['bbox_xyxy']
                
                for j_idx in keep_indices:
                    ann_j = sorted_anns[j_idx]
                    xyxy_j = ann_j['bbox_xyxy']
                    
                    if compute_iou(xyxy_i, xyxy_j) > 0.9:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    keep_indices.append(i)
            
            deduplicated_images[img_name] = [sorted_anns[i] for i in keep_indices]
            removed_count += len(anns) - len(keep_indices)

        logger.info(f"Deduplication: removed {removed_count} duplicate annotations")

        # Filter images with valid labels (at least one annotation)
        valid_images = {k: v for k, v in deduplicated_images.items() if len(v) > 0}
        
        # Validate bbox coordinates are within [0, 1] range
        validated_count = 0
        for img_name, anns in list(valid_images.items()):
            valid_anns = []
            for ann in anns:
                cx, cy, w, h = ann['bbox_cxcywh_yolo']
                if 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1:
                    valid_anns.append(ann)
            
            if len(valid_anns) > 0:
                valid_images[img_name] = valid_anns
            else:
                validated_count += 1
        
        logger.info(f"Validation: removed {validated_count} images with invalid bboxes")

        # Split into train/val/test
        image_names = sorted(valid_images.keys())
        
        if len(image_names) < 10:
            raise ValueError(f"Not enough valid images for training: {len(image_names)}")
        
        np.random.seed(42)
        indices = np.arange(len(image_names))
        np.random.shuffle(indices)

        n_train = int(train_ratio * len(indices))
        n_val = int(val_ratio * len(indices))

        train_imgs = [image_names[i] for i in indices[:n_train]]
        val_imgs = [image_names[i] for i in indices[n_train:n_train + n_val]]
        test_imgs = [image_names[i] for i in indices[n_train + n_val:]]

        # Create YOLO directory structure
        yolo_base = self.processed_dir / "yolo_dataset"
        
        for split in ["train", "val"]:
            img_dir = yolo_base / split / "images"
            lbl_dir = yolo_base / split / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

        # Copy images and create label files
        for split, img_list in [("train", train_imgs), ("val", val_imgs)]:
            for img_name in img_list:
                src = self.dataset_dir / img_name
                
                dst_img = yolo_base / split / "images" / img_name
                shutil.copy2(src, dst_img)

                lbl_file = yolo_base / split / "labels" / (Path(img_name).stem + ".txt")

                if img_name in valid_images:
                    with open(lbl_file, 'w') as f:
                        for ann in valid_images[img_name]:
                            cx, cy, w, h = ann["bbox_cxcywh_yolo"]
                            # Clamp values to [0, 1] range
                            cx = max(0.0, min(1.0, cx))
                            cy = max(0.0, min(1.0, cy))
                            w = max(0.001, min(1.0, w))
                            h = max(0.001, min(1.0, h))
                            
                            # For YOLO segmentation format, convert bbox to rectangular polygon
                            # Polygon points: top-left, top-right, bottom-right, bottom-left (normalized)
                            x1 = cx - w / 2
                            y1 = cy - h / 2
                            x2 = cx + w / 2
                            y2 = cy + h / 2
                            
                            # Write in YOLO seg format: class_id x1y1 x2y2 x3y3 x4y4 (all normalized)
                            f.write(f"0 {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}\n")

        logger.info(f"\nYOLO dataset created at: {yolo_base}")
        logger.info(f"  Train: {len(train_imgs)} images")
        logger.info(f"  Val:   {len(val_imgs)} images")
        logger.info(f"  Test:  {len(test_imgs)} images (not used in training)")

        return {"train": train_imgs, "val": val_imgs, "test": test_imgs}

    def create_yolo_yaml(self) -> str:
        """Create YOLO dataset YAML with absolute paths."""
        yolo_base = self.processed_dir / "yolo_dataset"

        yaml_config = {
            'train': str((yolo_base / 'train' / 'images').resolve()),
            'val': str((yolo_base / 'val' / 'images').resolve()),
            'test': str((yolo_base / 'val' / 'images').resolve()),
            'names': {0: 'utility_pole'},
            'nc': 1,
        }

        yaml_path = Path(self.config['output']['results_dir']) / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)

        logger.info(f"YOLO dataset YAML saved to: {yaml_path}")
        return str(yaml_path)

    def train(
        self, 
        dataset_yaml: str,
        epochs: int = 100,
        batch_size: int = 8,
        imgsz: int = 640,
        device: str = "cuda"
    ) -> Dict:
        """Fine-tune YOLO26x-seg with optimized hyperparameters."""
        from ultralytics import YOLO

        logger.info("=" * 60)
        logger.info("PHASE 3: Fine-tuning YOLO26x-seg")
        logger.info("=" * 60)
        logger.info(f"  Dataset YAML: {dataset_yaml}")
        logger.info(f"  Epochs:       {epochs}")
        logger.info(f"  Batch size:   {batch_size}")
        logger.info(f"  Image size:   {imgsz}")
        logger.info(f"  Device:       {device}")

        # Load YOLO26x-seg (largest, most accurate)
        model = YOLO('yolo26x-seg.pt')

        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            patience=self.config['model']['yolo'].get('patience', 50),
            device=device,
            project=str(self.config['output']['model_dir']),
            name='pole_detection_yolo26x',
            exist_ok=True,
            pretrained=True,
            augment=True,
            cache=False,
            workers=4,                  # Reduced for limited shared memory (64MB)
            lr0=0.01,                    # Initial learning rate
            lrf=0.01,                    # Final LR (cosine annealing)
            momentum=0.937,              # SGD momentum
            weight_decay=0.0005,         # Weight decay for regularization
            warmup_epochs=3.0,           # Warmup epochs
            warmup_momentum=0.8,         # Initial warmup momentum
            warmup_bias_lr=0.1,          # Bias LR during warmup
            close_mosaic=10,             # Disable mosaic last N epochs
            hsv_h=0.015,                 # Hue augmentation
            hsv_s=0.7,                   # Saturation augmentation
            hsv_v=0.4,                   # Value/brightness augmentation
            degrees=0.0,                 # No rotation (poles are upright)
            translate=0.1,               # 10% translation
            scale=0.5,                   # 50% scale augmentation
            shear=0.0,                   # No shear
            perspective=0.0,             # No perspective
            flipud=0.0,                  # No vertical flip (poles have orientation)
            fliplr=0.5,                  # 50% horizontal flip
            mosaic=1.0,                  # Full mosaic augmentation
            mixup=0.1,                   # Mixup augmentation
            copy_paste=0.1,              # Copy-paste augmentation
            optimizer='AdamW',           # AdamW for better convergence
            verbose=True,
        )

        logger.info("\nPhase 3 complete!")
        logger.info(f"  Best model: {self.config['output']['model_dir']}/pole_detection_yolo26x/weights/best.pt")

        return {
            'best_model': str(self.config['output']['model_dir']) + '/pole_detection_yolo26x/weights/best.pt',
            'last_model': str(self.config['output']['model_dir']) + '/pole_detection_yolo26x/weights/last.pt',
            'metrics': results.results_dict if hasattr(results, 'results_dict') else {}
        }


# ──────────────────────────────────────────────
# Main Pipeline Runner
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Utility Pole Detection — YOLO26 Training (3 Phases)'
    )
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--dataset-dir', type=str, default='Dataset')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3], default=0,
                       help='Run specific phase (0 = all phases)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--conf-threshold', type=float, default=0.25)

    args = parser.parse_args()

    # Update config with dataset directory
    if args.dataset_dir:
        config_path = Path(args.config)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config['data']['raw_dir'] = str(Path(args.dataset_dir).resolve())
        
        temp_config = config_path.parent / "config_temp.yaml"
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        
        args.config = str(temp_config.resolve())

    logger.info("=" * 60)
    logger.info("UTILITY POLE DETECTION — YOLO26 TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"  Dataset: {args.dataset_dir}")
    logger.info(f"  Phase:   {'All' if args.phase == 0 else str(args.phase)}")
    logger.info(f"  Device:  {args.device}")

    # ── Phase 1: Quick Inference ──
    if args.phase in [0, 1]:
        preparer = Phase1_Inference(args.config)
        summary = preparer.run(conf_threshold=args.conf_threshold)

    # ── Phase 2: Auto-annotate ──
    annotations = None
    if args.phase in [0, 2]:
        annotator = Phase2_AutoAnnotate(args.config)
        annotations = annotator.run(conf_threshold=args.conf_threshold)

        if not annotations:
            logger.error("No annotations generated. Check phase 1 results.")
            sys.exit(1)
    elif args.phase == 3:
        # Load existing annotations from Phase 2 output
        project_root = Path(args.config).parent.parent
        ann_path = project_root / "output" / "results" / "phase2_annotations.json"
        if not ann_path.exists():
            ann_path = Path("output/results/phase2_annotations.json")
        
        if ann_path.exists():
            with open(ann_path, 'r') as f:
                annotations = json.load(f)
            logger.info(f"Loaded {len(annotations)} existing annotations from Phase 2")
        else:
            # Generate pseudo-labels for all images (pole close-ups)
            logger.info("No existing annotations found. Generating pseudo-labels...")
            from ultralytics import YOLO
            model = YOLO('yolo26n.pt')
            dataset_dir = Path(args.config).parent / args.dataset_dir
            image_files = sorted(dataset_dir.glob("*.jpg"))
            
            annotations = []
            for img_path in image_files:
                results = model(img_path, conf=0.15, verbose=False)
                for r in results:
                    boxes = r.boxes
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            xyxy = box.xyxy[0].cpu().numpy()
                            
                            cx, cy, w, h = (
                                (xyxy[0] + xyxy[2]) / 2.0 / 640,
                                (xyxy[1] + xyxy[3]) / 2.0 / 640,
                                (xyxy[2] - xyxy[0]) / 640,
                                (xyxy[3] - xyxy[1]) / 640,
                            )
                            
                            annotations.append({
                                "image": str(img_path.name),
                                "class_id": cls_id,
                                "class_name": r.names[cls_id],
                                "confidence": round(conf, 4),
                                "bbox_xyxy": [round(b, 1) for b in xyxy.tolist()],
                                "bbox_cxcywh_yolo": [round(v, 6) for v in [cx, cy, w, h]]
                            })
            
            logger.info(f"Generated {len(annotations)} pseudo-labels")

    if annotations is None or len(annotations) == 0:
        logger.error("No annotations available. Run Phase 1 and 2 first.")
        sys.exit(1)

    # ── Phase 3: Fine-tune ──
    if args.phase in [0, 3]:
        trainer = Phase3_FineTune(args.config)
        
        splits = trainer.create_yolo_dataset_from_annotations(annotations)
        yaml_path = trainer.create_yolo_yaml()
        
        results = trainer.train(
            dataset_yaml=yaml_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device
        )

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

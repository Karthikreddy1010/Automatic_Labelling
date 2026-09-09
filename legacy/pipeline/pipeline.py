"""End-to-end pole detection and analysis pipeline.

This script orchestrates the full workflow:
1. Load images (from GSV or local directory)
2. Run YOLOv8 detection/segmentation
3. Extract attributes from masks
4. Evaluate against ground truth (if available)
5. Export results to CSV/JSON
"""

import os
import sys
import yaml
import json
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_preprocessing import DataPreprocessor, create_sample_dataset
from model_training import YOLOTrainer
from attribute_extraction import AttributeExtractor, PoleAttributes
from evaluation import DetectionMetrics, DetectionEvaluator, AttributeEvaluator, generate_evaluation_report

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PoleDetectionPipeline:
    """End-to-end pipeline for utility pole detection and analysis."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info("Initializing Pole Detection Pipeline")
        
        # Initialize components
        self.preprocessor = DataPreprocessor(self.config['data'])
        self.attribute_extractor = AttributeExtractor(self.config)
        
        # Paths
        self.output_dir = Path(self.config['output']['results_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_detection(
        self, 
        image_paths: List[str],
        model_path: Optional[str] = None,
        use_sample_model: bool = True
    ) -> List[PoleAttributes]:
        """Run pole detection on a list of images.
        
        Args:
            image_paths: Paths to input images
            model_path: Path to trained YOLO model (optional)
            use_sample_model: If True, create synthetic masks for demo
            
        Returns:
            List of PoleAttributes with detected poles and extracted attributes
        """
        
        from ultralytics import YOLO
        
        # Load or create model
        if model_path is None and use_sample_model:
            logger.info("Using sample/demo mode — generating synthetic detections")
            return self._run_demo_detection(image_paths)
        
        try:
            model = YOLO(model_path)
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Falling back to demo mode")
            return self._run_demo_detection(image_paths)
        
        all_attributes = []
        
        for i, img_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {img_path}")
            
            # Run detection
            results = model.predict(
                source=img_path,
                conf=0.25,  # Minimum confidence threshold
                iou=0.7,    # NMS IoU threshold
                imgsz=self.config['data']['image_width'],
                verbose=False
            )
            
            if results and len(results) > 0:
                result = results[0]
                
                # Process each detected pole in the image
                if result.masks is not None and len(result.masks) > 0:
                    for j, mask in enumerate(result.masks):
                        # Get mask as numpy array
                        mask_tensor = mask.data.cpu().numpy()
                        
                        # Extract attributes
                        metadata = {
                            'filepath': img_path,
                            'confidence': float(result.conf[j]) if result.conf is not None else 0.0,
                            'latitude': self.config['data'].get('default_lat', 0.0),
                            'longitude': self.config['data'].get('default_lon', 0.0),
                        }
                        
                        attrs = self.attribute_extractor.extract_from_mask(
                            mask_tensor.astype(np.uint8) * 255,
                            None,  # Image will be loaded internally if needed
                            metadata,
                            pole_id=i * 100 + j
                        )
                        
                        all_attributes.append(attrs)
        
        logger.info(f"Detection complete: {len(all_attributes)} poles detected")
        return all_attributes
    
    def _run_demo_detection(
        self, 
        image_paths: List[str]
    ) -> List[PoleAttributes]:
        """Run demo detection using synthetic masks on sample images."""
        
        import cv2
        import numpy as np
        
        all_attributes = []
        
        for i, img_path in enumerate(image_paths):
            # Load the image to get dimensions
            try:
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning(f"Could not load {img_path}, skipping")
                    continue
                h, w = img.shape[:2]
            except Exception as e:
                logger.warning(f"Error loading image {img_path}: {e}")
                continue
            
            # Create synthetic pole mask (centered vertical rectangle)
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Simulate a pole in the center-left area
            pole_x = int(w * 0.35)
            pole_w = int(w * 0.02)
            pole_top = int(h * 0.15)
            pole_bottom = int(h * 0.85)
            
            cv2.rectangle(mask, 
                         (pole_x - pole_w // 2, pole_top),
                         (pole_x + pole_w // 2, pole_bottom),
                         255, -1)
            
            # Add slight tilt for realism
            if i % 3 == 0:
                # Create tilted mask
                M = cv2.getRotationMatrix2D((w//2, h//2), 3, 1)
                mask = cv2.warpAffine(mask, M, (w, h))
            
            metadata = {
                'filepath': img_path,
                'confidence': 0.95,
                'latitude': 40.8276,  # Montclair, NJ (project location)
                'longitude': -74.2143,
                'heading': np.random.uniform(0, 360),
                'pitch': np.random.uniform(-5, 5),
            }
            
            attrs = self.attribute_extractor.extract_from_mask(
                mask, img, metadata, pole_id=i
            )
            
            all_attributes.append(attrs)
        
        logger.info(f"Demo detection complete: {len(all_attributes)} poles detected")
        return all_attributes
    
    def export_results(
        self, 
        attributes: List[PoleAttributes],
        format: str = 'csv'
    ) -> str:
        """Export detection results to file."""
        
        if not attributes:
            logger.warning("No attributes to export")
            return ""
        
        df = self.attribute_extractor.to_dataframe(attributes)
        
        if format == 'csv':
            output_path = self.output_dir / "pole_detection_results.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Results exported to {output_path}")
            
        elif format == 'json':
            records = []
            for attr in attributes:
                record = {
                    'image_path': attr.image_path,
                    'pole_id': attr.pole_id,
                    'confidence': float(attr.confidence),
                    'bounding_box': list(attr.bounding_box),
                    'mask_area_pixels': int(attr.mask_area_pixels),
                    'height_meters': attr.height_meters,
                    'tilt_corrected_degrees': attr.tilt_corrected_degrees,
                    'latitude': attr.latitude,
                    'longitude': attr.longitude,
                }
                records.append(record)
            
            output_path = self.output_dir / "pole_detection_results.json"
            with open(output_path, 'w') as f:
                json.dump(records, f, indent=2)
            logger.info(f"Results exported to {output_path}")
        
        return str(self.output_dir / f"pole_detection_results.{format}")
    
    def run_evaluation(
        self,
        attributes: List[PoleAttributes],
        ground_truth_heights: Optional[List[float]] = None,
        ground_truth_tilts: Optional[List[float]] = None,
        ground_truth_boxes: Optional[List[Dict]] = None,
    ) -> str:
        """
        Run evaluation and generate report.

        Each metric is only computed when its ground truth is actually
        supplied by the caller -- there is no synthetic/placeholder fallback.
        A metric whose ground truth is missing is reported as "not
        evaluated" rather than a fabricated number.

        Args:
            ground_truth_boxes: one dict per ground-truth box, each with a
                'bbox' key in (x1, y1, x2, y2) pixel format, aligned to the
                same images as `attributes`. Required to compute detection
                precision/recall/F1/mAP.
        """
        detection_metrics = None
        if ground_truth_boxes:
            predictions = [
                {
                    'bbox': (
                        a.bounding_box[0],
                        a.bounding_box[1],
                        a.bounding_box[0] + a.bounding_box[2],
                        a.bounding_box[1] + a.bounding_box[3],
                    ),
                    'confidence': a.confidence,
                }
                for a in attributes
            ]
            detection_metrics = DetectionEvaluator.evaluate_detection(
                predictions, ground_truth_boxes
            )

        # Attribute evaluation
        height_errors = None
        if ground_truth_heights:
            predicted_heights = [a.height_meters for a in attributes if a.height_meters is not None]
            height_errors = AttributeEvaluator.compute_height_error(
                predicted_heights, ground_truth_heights
            )

        tilt_errors = None
        if ground_truth_tilts:
            predicted_tilts = [a.tilt_corrected_degrees for a in attributes]
            tilt_errors = AttributeEvaluator.compute_tilt_error(
                predicted_tilts, ground_truth_tilts
            )

        # Generate report
        report = generate_evaluation_report(
            detection_metrics, height_errors, tilt_errors
        )
        
        # Save report
        report_path = self.output_dir / "evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Evaluation report saved to {report_path}")
        print(report)
        
        return str(report_path)


def main():
    """Main entry point for the pipeline."""
    
    parser = argparse.ArgumentParser(
        description='Utility Pole Detection Pipeline'
    )
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--images-dir', type=str, 
                       default='data/raw',
                       help='Directory containing images')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained YOLO model')
    parser.add_argument('--demo', action='store_true',
                       help='Run in demo mode with synthetic data')
    parser.add_argument('--generate-samples', type=int, default=0,
                       help='Generate N sample images for testing')
    parser.add_argument('--export-format', type=str, 
                       choices=['csv', 'json'], default='csv',
                       help='Export format for results')
    
    args = parser.parse_args()
    
    # Generate sample images if requested
    if args.generate_samples > 0:
        logger.info(f"Generating {args.generate_samples} sample images...")
        create_sample_dataset(args.images_dir, args.generate_samples)
    
    # Initialize pipeline
    pipeline = PoleDetectionPipeline(args.config)
    
    # Get image paths
    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        logger.error(f"Images directory {images_dir} does not exist")
        sys.exit(1)
    
    image_paths = sorted([str(p) for p in images_dir.glob("*.jpg")])
    
    if not image_paths:
        logger.error(f"No .jpg files found in {images_dir}")
        print("\nTo generate sample data:")
        print(f"  python pipeline.py --generate-samples 100")
        sys.exit(1)
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    # Run detection
    attributes = pipeline.run_detection(
        image_paths, 
        model_path=args.model_path,
        use_sample_model=args.demo
    )
    
    if not attributes:
        logger.warning("No poles detected. Check confidence threshold or try demo mode.")
        sys.exit(1)
    
    # Export results
    output_file = pipeline.export_results(attributes, args.export_format)
    
    # Run evaluation
    report_path = pipeline.run_evaluation(attributes)
    
    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"Results: {output_file}")
    print(f"Report:  {report_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

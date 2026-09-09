"""Evaluation metrics for pole detection and attribute estimation."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectionMetrics:
    """Container for detection evaluation metrics."""
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    map_50: float = 0.0
    map_50_95: float = 0.0
    mAP_per_class: Dict[str, float] = None
    
    def __post_init__(self):
        if self.mAP_per_class is None:
            self.mAP_per_class = {}


class DetectionEvaluator:
    """Evaluate pole detection performance."""
    
    @staticmethod
    def compute_iou(box1: Tuple[float, float, float, float], 
                    box2: Tuple[float, float, float, float]) -> float:
        """Compute Intersection over Union between two bounding boxes.
        
        Args:
            box1: (x1, y1, x2, y2) format
            box2: (x1, y1, x2, y2) format
            
        Returns:
            IoU value between 0 and 1
        """
        # Compute intersection coordinates
        x_inter = max(box1[0], box2[0])
        y_inter = max(box1[1], box2[1])
        x_end = min(box1[2], box2[2])
        y_end = min(box1[3], box2[3])
        
        # Check for no intersection
        if x_end < x_inter or y_end < y_inter:
            return 0.0
        
        inter_area = (x_end - x_inter) * (y_end - y_inter)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
            
        return inter_area / union_area
    
    @staticmethod
    def compute_precision_recall(
        predictions: List[Dict], 
        ground_truths: List[Dict],
        iou_threshold: float = 0.5
    ) -> Tuple[float, float]:
        """Compute precision and recall for detection results.
        
        Args:
            predictions: List of dicts with 'bbox', 'confidence' keys
            ground_truths: List of dicts with 'bbox' key
            iou_threshold: IoU threshold for matching
            
        Returns:
            (precision, recall) tuple
        """
        if not predictions or not ground_truths:
            return 0.0, 0.0
        
        # Sort predictions by confidence (descending)
        sorted_preds = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        tp = 0  # True positives
        fp = 0  # False positives
        matched_gt = set()
        
        for pred in sorted_preds:
            best_iou = 0.0
            best_idx = -1
            
            for i, gt in enumerate(ground_truths):
                if i in matched_gt:
                    continue
                    
                iou = DetectionEvaluator.compute_iou(
                    pred['bbox'], gt['bbox']
                )
                
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            if best_iou >= iou_threshold and best_idx != -1:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / len(ground_truths) if ground_truths else 0.0
        
        return round(precision, 4), round(recall, 4)
    
    @staticmethod
    def compute_f1(precision: float, recall: float) -> float:
        """Compute F1 score from precision and recall."""
        if precision + recall == 0:
            return 0.0
        return round(2 * precision * recall / (precision + recall), 4)
    
    @staticmethod
    def evaluate_detection(
        predictions: List[Dict], 
        ground_truths: List[Dict]
    ) -> DetectionMetrics:
        """Full detection evaluation."""
        
        precisions = []
        recalls = []
        
        # Evaluate at multiple IoU thresholds (for mAP)
        for iou_thresh in [0.5, 0.75]:
            p, r = DetectionEvaluator.compute_precision_recall(
                predictions, ground_truths, iou_threshold=iou_thresh
            )
            precisions.append(p)
            recalls.append(r)
        
        # Use mAP@0.5 as primary metric
        precision = precisions[0]
        recall = recalls[0]
        f1 = DetectionEvaluator.compute_f1(precision, recall)
        
        metrics = DetectionMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            map_50=round((precision + recall) / 2 * f1, 4),
            map_50_95=round(sum(precisions) / len(precisions) * 
                          sum(recalls) / len(recalls) * f1, 4)
        )
        
        logger.info(f"Detection Metrics: P={precision:.3f}, R={recall:.3f}, "
                   f"F1={f1:.3f}")
        
        return metrics


class AttributeEvaluator:
    """Evaluate attribute estimation accuracy."""
    
    @staticmethod
    def compute_height_error(
        predicted_heights: List[float], 
        true_heights: List[float]
    ) -> Dict[str, float]:
        """Compute height estimation errors.
        
        Returns dict with MAE, RMSE, and percentage of predictions within tolerance.
        """
        if not predicted_heights or not true_heights:
            return {'mae': 0.0, 'rmse': 0.0, 'within_tolerance_1m': 0.0}
        
        errors = [abs(p - t) for p, t in zip(predicted_heights, true_heights)]
        
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        
        # Count predictions within acceptable tolerance (1 meter)
        within_tolerance = sum(1 for e in errors if e <= 1.0)
        tolerance_rate = within_tolerance / len(errors)
        
        return {
            'mae': round(float(mae), 4),
            'rmse': round(float(rmse), 4),
            'within_tolerance_1m': round(tolerance_rate, 4)
        }
    
    @staticmethod
    def compute_tilt_error(
        predicted_tilts: List[float], 
        true_tilts: List[float]
    ) -> Dict[str, float]:
        """Compute tilt angle estimation errors."""
        
        if not predicted_tilts or not true_tilts:
            return {'mae': 0.0, 'rmse': 0.0, 'within_tolerance_5deg': 0.0}
        
        errors = [abs(p - t) for p, t in zip(predicted_tilts, true_tilts)]
        
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        
        within_tolerance = sum(1 for e in errors if e <= 5.0)
        tolerance_rate = within_tolerance / len(errors)
        
        return {
            'mae': round(float(mae), 4),
            'rmse': round(float(rmse), 4),
            'within_tolerance_5deg': round(tolerance_rate, 4)
        }


def generate_evaluation_report(
    detection_metrics: Optional[DetectionMetrics],
    height_errors: Optional[Dict[str, float]],
    tilt_errors: Optional[Dict[str, float]]
) -> str:
    """
    Generate a formatted evaluation report.

    Each section is computed only when its ground truth was actually
    supplied; a section whose ground truth is missing is reported as
    "not evaluated" rather than filled in with a placeholder number.
    """
    if detection_metrics is not None:
        detection_section = (
            f"  Precision:     {detection_metrics.precision:.4f}\n"
            f"  Recall:        {detection_metrics.recall:.4f}\n"
            f"  F1 Score:      {detection_metrics.f1_score:.4f}\n"
            f"  mAP@50:        {detection_metrics.map_50:.4f}\n"
            f"  mAP@50-95:     {detection_metrics.map_50_95:.4f}"
        )
    else:
        detection_section = "  Not evaluated -- no ground-truth boxes were supplied."

    if height_errors is not None:
        height_section = (
            f"  MAE (meters):      {height_errors['mae']:.4f}\n"
            f"  RMSE (meters):     {height_errors['rmse']:.4f}\n"
            f"  Within ±1m:        {height_errors['within_tolerance_1m']*100:.1f}%"
        )
    else:
        height_section = "  Not evaluated -- no ground-truth heights were supplied."

    if tilt_errors is not None:
        tilt_section = (
            f"  MAE (degrees):     {tilt_errors['mae']:.4f}\n"
            f"  RMSE (degrees):    {tilt_errors['rmse']:.4f}\n"
            f"  Within ±5°:        {tilt_errors['within_tolerance_5deg']*100:.1f}%"
        )
    else:
        tilt_section = "  Not evaluated -- no ground-truth tilts were supplied."

    report = f"""
{'='*60}
  POLE DETECTION & ATTRIBUTE ESTIMATION EVALUATION REPORT
{'='*60}

DETECTION PERFORMANCE:
{detection_section}

HEIGHT ESTIMATION:
{height_section}

TILT ESTIMATION:
{tilt_section}

{'='*60}
"""

    return report

"""Attribute extraction from segmented pole masks.

Estimates height (meters), tilt angle (degrees), and width from 
segmentation masks using camera geometry and reference objects.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class PoleAttributes:
    """Container for extracted pole attributes."""
    image_path: str = ""
    pole_id: int = 0
    confidence: float = 0.0
    
    # Detection
    bounding_box: Tuple[float, float, float, float] = (0, 0, 0, 0)  # x,y,w,h
    mask_area_pixels: int = 0
    
    # Height estimation
    height_pixels: int = 0
    height_meters: Optional[float] = None
    height_method: str = ""  # "reference" or "camera_geometry"
    
    # Tilt estimation  
    tilt_image_degrees: float = 0.0
    tilt_corrected_degrees: float = 0.0
    
    # Width estimation
    width_pixels: int = 0
    width_meters: Optional[float] = None
    
    # Metadata
    latitude: float = 0.0
    longitude: float = 0.0
    heading: float = 0.0
    pitch: float = 0.0
    pano_id: str = ""


class AttributeExtractor:
    """Extract geometric attributes from pole segmentation masks."""
    
    def __init__(self, config: Dict):
        self.config = config
        attr_config = config.get('attribute_extraction', {})
        
        # Reference object heights (meters)
        ref = attr_config.get('reference_objects', {})
        self.reference_heights = {
            'human': ref.get('human_height', 1.7),
            'stop_sign': ref.get('stop_sign_height', 2.5),
            'vehicle': ref.get('vehicle_height', 1.5)
        }
        
        # Camera parameters
        cam = attr_config.get('camera', {})
        self.camera_height = config['data'].get('camera_height', 2.5)
        self.focal_length_px = cam.get('focal_length_pixels')
        self.sensor_width_mm = cam.get('sensor_width_mm', 6.4)
        
    def extract_from_mask(
        self, 
        mask: np.ndarray, 
        image: np.ndarray,
        metadata: Dict = None,
        pole_id: int = 0
    ) -> PoleAttributes:
        """Extract all attributes from a single pole segmentation mask."""
        
        if metadata is None:
            metadata = {}
            
        attrs = PoleAttributes(
            image_path=metadata.get('filepath', ''),
            pole_id=pole_id,
            confidence=metadata.get('confidence', 1.0),
            latitude=metadata.get('latitude', 0.0),
            longitude=metadata.get('longitude', 0.0),
            heading=metadata.get('heading', 0.0),
            pitch=metadata.get('pitch', 0.0),
            pano_id=metadata.get('pano_id', '')
        )
        
        # --- Bounding box and mask area ---
        coords = cv2.findNonZero(mask.astype(np.uint8))
        if coords is None:
            logger.warning(f"No pole pixels found in mask for pole {pole_id}")
            return attrs
            
        x, y, w, h = cv2.boundingRect(coords)
        attrs.bounding_box = (float(x), float(y), float(w), float(h))
        attrs.mask_area_pixels = int(np.sum(mask > 0))
        
        # --- Height estimation ---
        attrs.height_pixels = h
        
        # Method 1: Camera geometry (pinhole model)
        height_m, method = self._estimate_height_camera_geometry(
            mask, image, metadata
        )
        if height_m is not None:
            attrs.height_meters = round(height_m, 2)
            attrs.height_method = "camera_geometry"
        
        # Method 2: Reference object scaling (if available)
        ref_height = self._estimate_height_reference_object(
            mask, image, metadata
        )
        if ref_height is not None and attrs.height_meters is None:
            attrs.height_meters = round(ref_height, 2)
            attrs.height_method = "reference"
        
        # --- Tilt estimation ---
        tilt_image, tilt_corrected = self._estimate_tilt(mask, metadata)
        attrs.tilt_image_degrees = round(tilt_image, 2)
        attrs.tilt_corrected_degrees = round(tilt_corrected, 2)
        
        # --- Width estimation ---
        attrs.width_pixels = w
        
        return attrs
    
    def _estimate_height_camera_geometry(
        self, 
        mask: np.ndarray, 
        image: np.ndarray,
        metadata: Dict
    ) -> Tuple[Optional[float], str]:
        """Estimate pole height using pinhole camera model.
        
        Formula: H_real = (h_pixels * D) / f
        
        Where:
            h_pixels = pixel height of the pole in the mask
            D = distance from camera to pole (estimated from metadata)
            f = focal length in pixels
            
        For GSV images, we approximate distance using the pitch angle 
        and camera height.
        """
        
        if self.focal_length_px is None:
            # Estimate focal length from image dimensions assuming ~90° FOV
            img_width = image.shape[1] if image is not None else 640
            fov_degrees = metadata.get('fov', 90)
            self.focal_length_px = (img_width / 2) / np.tan(
                np.radians(fov_degrees / 2)
            )
        
        h_pixels = mask.shape[0]  # Height of the pole in pixels
        
        # Estimate distance to pole from camera pitch and height
        pitch_rad = np.radians(metadata.get('pitch', 0.0))
        
        if abs(pitch_rad) < 0.05:  # Camera is roughly level (< 3 degrees)
            # Use typical GSV scene depth estimate for street-level imagery
            D_estimate = 15.0  # meters (typical distance to roadside objects)
        else:
            # Approximate: D ≈ camera_height / tan(|pitch|)
            D_estimate = self.camera_height / np.tan(abs(pitch_rad))
        
        if D_estimate <= 0 or self.focal_length_px <= 0:
            return None, "camera_geometry"
        
        height_meters = (h_pixels * D_estimate) / self.focal_length_px
        
        # Clamp to realistic utility pole heights (3-25 meters)
        # This handles cases where the mask covers most of the image
        if height_meters < 3:
            logger.warning(
                f"Height estimate {height_meters:.1f}m is below minimum. "
                f"h_pixels={h_pixels}, D={D_estimate:.1f}m, f={self.focal_length_px:.0f}px"
            )
            height_meters = max(height_meters, 3.0)
        elif height_meters > 25:
            logger.warning(
                f"Height estimate {height_meters:.1f}m exceeds typical pole range. "
                f"Clamping to realistic maximum of 25m. "
                f"h_pixels={h_pixels}, D={D_estimate:.1f}m, f={self.focal_length_px:.0f}px"
            )
            height_meters = min(height_meters, 25.0)
        
        return round(height_meters, 2), "camera_geometry"
    
    def _estimate_height_reference_object(
        self, 
        mask: np.ndarray, 
        image: np.ndarray,
        metadata: Dict
    ) -> Optional[float]:
        """Estimate pole height using a reference object in the scene.
        
        Uses similar triangles: H_pole / h_pole_pixels = H_ref / h_ref_pixels
        
        For now, returns None since we'd need to detect a reference object 
        (human, STOP sign, vehicle) in the same image. This is a placeholder 
        for future integration with an object detector.
        """
        
        # TODO: Integrate with a human/vehicle detector to find reference objects
        # For now, use a default reference height assumption
        
        return None
    
    def _estimate_tilt(
        self, 
        mask: np.ndarray, 
        metadata: Dict
    ) -> Tuple[float, float]:
        """Estimate pole tilt angle from segmentation mask.
        
        1. Fit a line through the center of mass of the mask to get apparent tilt
        2. Correct for camera pitch using GSV metadata
        
        Returns: (apparent_tilt_degrees, corrected_tilt_degrees)
        """
        
        # Get all non-zero pixel coordinates from the mask
        y_coords, x_coords = np.where(mask > 0)
        
        if len(x_coords) < 10:
            return 0.0, 0.0
        
        # Compute center of mass
        cx = np.mean(x_coords)
        cy = np.mean(y_coords)
        
        # Fit a line through the mask pixels using least squares
        # y = mx + c (in image coordinates, y increases downward)
        if len(x_coords) > 1:
            coeffs = np.polyfit(x_coords, y_coords, 1)
            slope = coeffs[0]  # dy/dx in pixel space
            
            # Convert to angle from vertical
            # In image coords: positive slope means leaning right (top is left of bottom)
            tilt_image_degrees = np.degrees(np.arctan(slope))
        else:
            tilt_image_degrees = 0.0
        
        # Correct for camera pitch
        camera_pitch = metadata.get('pitch', 0.0)
        
        # The corrected tilt is the apparent tilt minus the camera's vertical angle
        # If camera pitched up by α, objects appear tilted down by α
        tilt_corrected_degrees = tilt_image_degrees - camera_pitch
        
        return round(tilt_image_degrees, 2), round(tilt_corrected_degrees, 2)
    
    def extract_batch(
        self, 
        masks: List[np.ndarray], 
        images: List[np.ndarray],
        metadatas: List[Dict] = None,
        start_id: int = 0
    ) -> List[PoleAttributes]:
        """Extract attributes from a batch of pole detections."""
        
        if metadatas is None:
            metadatas = [{} for _ in masks]
            
        results = []
        for i, (mask, image, metadata) in enumerate(
            zip(masks, images, metadatas)
        ):
            attrs = self.extract_from_mask(
                mask, image, metadata, pole_id=start_id + i
            )
            results.append(attrs)
            
        return results
    
    def to_dataframe(self, attributes: List[PoleAttributes]) -> 'pandas.DataFrame':
        """Convert list of PoleAttributes to a pandas DataFrame for export."""
        
        import pandas as pd
        
        records = []
        for attr in attributes:
            record = {
                'image_path': attr.image_path,
                'pole_id': attr.pole_id,
                'confidence': attr.confidence,
                'bbox_x': attr.bounding_box[0],
                'bbox_y': attr.bounding_box[1],
                'bbox_w': attr.bounding_box[2],
                'bbox_h': attr.bounding_box[3],
                'mask_area_pixels': attr.mask_area_pixels,
                'height_pixels': attr.height_pixels,
                'height_meters': attr.height_meters,
                'height_method': attr.height_method,
                'tilt_image_degrees': attr.tilt_image_degrees,
                'tilt_corrected_degrees': attr.tilt_corrected_degrees,
                'width_pixels': attr.width_pixels,
                'latitude': attr.latitude,
                'longitude': attr.longitude,
                'heading': attr.heading,
                'pitch': attr.pitch,
                'pano_id': attr.pano_id
            }
            records.append(record)
        
        return pd.DataFrame(records)

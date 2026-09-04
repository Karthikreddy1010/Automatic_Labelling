"""GSV Data Collection module for fetching street view images."""

import os
import time
import requests
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class GSVCollector:
    """Collects Google Street View images and metadata."""
    
    def __init__(self, api_key: str, output_dir: str = "data/raw"):
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_image_url(
        self, 
        lat: float, 
        lon: float, 
        heading: float, 
        pitch: float = 0.0, 
        fov: int = 90,
        size: str = "640x640"
    ) -> str:
        """Construct GSV static image URL."""
        base_url = "https://maps.googleapis.com/maps/api/streetview"
        params = {
            "location": f"{lat},{lon}",
            "heading": heading,
            "pitch": pitch,
            "fov": fov,
            "size": size,
            "key": self.api_key
        }
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{base_url}?{query_string}"
    
    def download_image(
        self, 
        lat: float, 
        lon: float, 
        heading: float, 
        pitch: float = 0.0, 
        fov: int = 90,
        size: str = "640x640",
        max_retries: int = 3
    ) -> Optional[Tuple[bytes, Dict]]:
        """Download a single GSV image with metadata."""
        url = self.get_image_url(lat, lon, heading, pitch, fov, size)
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    metadata = {
                        "latitude": lat,
                        "longitude": lon,
                        "heading": heading,
                        "pitch": pitch,
                        "fov": fov,
                        "size": size,
                        "pano_id": self._extract_pano_id(url),
                        "timestamp": time.time()
                    }
                    return response.content, metadata
                    
                elif response.status_code == 403:
                    logger.warning("API key invalid or quota exceeded")
                    return None, {}
                    
                elif response.status_code == 503:
                    # No imagery available at this location
                    logger.info(f"No GSV imagery at ({lat}, {lon}) heading={heading}")
                    return None, {}
                    
                else:
                    wait_time = (attempt + 1) * 2
                    logger.warning(
                        f"Request failed with status {response.status_code}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    
        return None, {}
    
    def _extract_pano_id(self, url: str) -> Optional[str]:
        """Extract panorama ID from GSV URL."""
        params = dict(p.split("=") for p in url.split("?")[1].split("&"))
        return params.get("pano")
    
    def collect_around_location(
        self, 
        lat: float, 
        lon: float, 
        headings: List[float], 
        pitch: float = 0.0,
        fov: int = 90
    ) -> List[Dict]:
        """Collect images around a single GPS location at multiple headings."""
        results = []
        
        for heading in headings:
            image_data, metadata = self.download_image(
                lat, lon, heading, pitch, fov
            )
            
            if image_data is not None:
                # Save image with descriptive filename
                safe_heading = str(int(heading)).replace("-", "neg")
                filename = f"lat{lat:.4f}_lon{lon:.4f}_h{safe_heading}.jpg"
                filepath = self.output_dir / filename
                
                with open(filepath, "wb") as f:
                    f.write(image_data)
                    
                metadata["filepath"] = str(filepath)
                results.append(metadata)
                
            time.sleep(0.1)  # Rate limiting
            
        return results
    
    def collect_from_locations(self, locations: List[Dict]) -> Dict[str, int]:
        """Collect images from multiple GPS coordinates.
        
        Args:
            locations: List of dicts with keys: lat, lon, headings (list), 
                      pitch (optional), fov (optional)
                      
        Returns:
            Summary dict with counts of successful/failed collections
        """
        total_success = 0
        total_failed = 0
        
        for i, loc in enumerate(locations):
            logger.info(
                f"Collecting images {i+1}/{len(locations)}: "
                f"({loc['lat']}, {loc['lon']})"
            )
            
            headings = loc.get("headings", [0, 90, 180, 270])
            pitch = loc.get("pitch", 0.0)
            fov = loc.get("fov", 90)
            
            results = self.collect_around_location(
                loc["lat"], loc["lon"], headings, pitch, fov
            )
            
            total_success += len(results)
            total_failed += len(headings) - len(results)
            
        logger.info(f"Collection complete: {total_success} success, {total_failed} failed")
        return {"success": total_success, "failed": total_failed}

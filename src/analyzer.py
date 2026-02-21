import os
import cv2
import json
import logging
import numpy as np
import uuid
from typing import Dict, Any, List
from PIL import Image

from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection
from surya.model.detection.model import load_model, load_processor
from surya.ocr import run_ocr
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from surya.settings import settings

logger = logging.getLogger(__name__)

class SlideAnalyzer:
    def __init__(self, output_dir: str = "/tmp/pdf_staging"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        logger.info("Initializing Surya Layout and OCR models...")
        self.det_model = load_model()
        self.det_processor = load_processor()
        
        # Surya layout models use the detection architecture bindings
        self.layout_model = load_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
        self.layout_processor = load_processor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
        
        self.rec_model = load_rec_model()
        self.rec_processor = load_rec_processor()
        
    def _extract_background_color(self, img_bgr: np.ndarray, bboxes: List[List[int]]) -> str:
        """Finds the most dominant background color by masking out layout boxes."""
        mask = np.ones(img_bgr.shape[:2], dtype=np.uint8) * 255
        
        # Mask out content
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
            
        bg_pixels = img_bgr[mask == 255]
        
        if len(bg_pixels) == 0:
            return "#FFFFFF" # Fallback if image is entirely filled
            
        # Reshape and find the most frequent color
        bg_pixels = bg_pixels.reshape(-1, 3)
        colors, counts = np.unique(bg_pixels, axis=0, return_counts=True)
        dominant_color = colors[np.argmax(counts)]
        
        # BGR to RGB to Hex
        dominant_rgb = dominant_color[::-1]
        hex_color = "#{:02x}{:02x}{:02x}".format(dominant_rgb[0], dominant_rgb[1], dominant_rgb[2])
        return hex_color

    def _get_intersection(self, box1: List[int], box2: List[int]) -> float:
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        return (x_right - x_left) * (y_bottom - y_top)

    def _split_into_sub_blocks(self, img_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int, block_label: str) -> List[List[int]]:
        # Do not run morphological splitting on semantic text logic groups. Surya handles text structures well.
        if block_label in ["Text", "Title", "List", "Caption", "Formula"]:
            return [[x1, y1, x2, y2]]
            
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0: return [[x1, y1, x2, y2]]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        is_figure = block_label in ["Figure", "Picture"]
        if is_figure:
            panel_kernel_size = max(10, min(x2-x1, y2-y1) // 30) 
            kernel = np.ones((panel_kernel_size, panel_kernel_size), np.uint8)
        else:
            kernel = np.ones((15, 25), np.uint8) 

        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_panels = []
        area_thresh = ((x2-x1)*(y2-y1)) * 0.02
        for c in contours:
            cx, cy, cw, ch = cv2.boundingRect(c)
            # Filter noise
            if cw * ch > area_thresh and cw > 15 and ch > 15:
                valid_panels.append([x1+cx, y1+cy, x1+cx+cw, y1+cy+ch])
        
        if len(valid_panels) > 1:
            valid_panels.sort(key=lambda b: (b[1] // 50, b[0]))
            return valid_panels
            
        return [[x1, y1, x2, y2]]

    def analyze_slide(self, image_path: str, slide_index: int) -> Dict[str, Any]:
        """
        Analyzes a single slide image.
        Returns a dictionary chunk ready to be appended to the JSON IR's `slides` list.
        """
        logger.info(f"Analyzing Slide {slide_index}: {image_path}")
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
            
        img_pil = Image.open(image_path).convert("RGB")
        
        # 1. Detect text lines
        line_predictions = batch_text_detection([img_pil], self.det_model, self.det_processor)
        
        # 2. Detect layout logic blocks
        layout_predictions = batch_layout_detection([img_pil], self.layout_model, self.layout_processor, line_predictions)
        layout_result = layout_predictions[0]
        
        # 3. Perform full-page OCR
        ocr_predictions = run_ocr([img_pil], [["en"]], self.det_model, self.det_processor, self.rec_model, self.rec_processor)
        ocr_result = ocr_predictions[0]
        
        slide_output = {
            "slide_index": slide_index,
            "background_color": "#FFFFFF",
            "elements": []
        }
        
        all_bboxes = []
        
        def get_bbox(polygon):
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
            
        ocr_lines = []
        for line in ocr_result.text_lines:
            ocr_lines.append({"bbox": get_bbox(line.polygon), "text": line.text})
            
        for block_idx, block in enumerate(layout_result.bboxes):
            x1, y1, x2, y2 = get_bbox(block.polygon)
            block_type = block.label
            
            sub_blocks = self._split_into_sub_blocks(img_bgr, x1, y1, x2, y2, block_type)
            
            for sub_idx, sb in enumerate(sub_blocks):
                sb_x1, sb_y1, sb_x2, sb_y2 = sb
                sb_area = max(1, (sb_x2 - sb_x1) * (sb_y2 - sb_y1))
                all_bboxes.append(sb)
                
                # Intersect text
                block_text = []
                text_area_covered = 0
                
                for line in ocr_lines:
                    l_bbox = line["bbox"]
                    inter = self._get_intersection(sb, l_bbox)
                    l_area = (l_bbox[2]-l_bbox[0]) * (l_bbox[3]-l_bbox[1])
                    
                    if l_area > 0 and inter / l_area > 0.4:
                        block_text.append(line["text"])
                        text_area_covered += inter
                
                text_coverage_ratio = text_area_covered / sb_area
                w = sb_x2 - sb_x1
                h = sb_y2 - sb_y1
                
                if block_type in ["Text", "Title", "List", "Caption", "Formula"] or (block_type == "Table" and len(block_text) > 0 and text_coverage_ratio > 0.05):
                    raw_text = " ".join(block_text).strip()
                    if raw_text:
                        element = {
                            "type": "title" if block_type == "Title" else "text",
                            "text": raw_text,
                            "bbox": [sb_x1, sb_y1, w, h],
                            "estimated_size": max(12, int(h / max(1, len(block_text)))),
                            "is_bold": True if block_type == "Title" else False
                        }
                        slide_output["elements"].append(element)
                        
                elif block_type in ["Figure", "Picture", "Table"]:
                    if w <= 50 or h <= 50:
                        continue
                        
                    crop_bgr = img_bgr[sb_y1:sb_y2, sb_x1:sb_x2]
                    base_name = os.path.basename(image_path).split('.')[0]
                    fig_name = f"{base_name}_fig_{block_idx}_{sub_idx}.png"
                    fig_path = os.path.join(self.output_dir, fig_name)
                    
                    cv2.imwrite(fig_path, crop_bgr)
                    
                    element = {
                        "type": "figure",
                        "source_file": fig_path,
                        "bbox": [sb_x1, sb_y1, w, h]
                    }
                    slide_output["elements"].append(element)
                
        # 4. Estimate background color
        try:
            bg_color = self._extract_background_color(img_bgr, all_bboxes)
            slide_output["background_color"] = bg_color
        except Exception as e:
            logger.warning(f"Failed to extract background color for {image_path}: {e}")
            
        return slide_output

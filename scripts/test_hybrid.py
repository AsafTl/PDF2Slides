import cv2
import json
import numpy as np
import os
from PIL import Image

from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection
from surya.ocr import run_ocr
from src.analyzer import SlideAnalyzer

def get_intersection(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    return (x_right - x_left) * (y_bottom - y_top)

def get_bbox(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

def test_hybrid():
    analyzer = SlideAnalyzer(output_dir='tests/craters_staging')
    img_path = 'tests/craters/page_004.png'
    img_bgr = cv2.imread(img_path)
    img_pil = Image.open(img_path).convert("RGB")
    
    print("Running Surya predictions...")
    line_predictions = batch_text_detection([img_pil], analyzer.det_model, analyzer.det_processor)
    layout_predictions = batch_layout_detection([img_pil], analyzer.layout_model, analyzer.layout_processor, line_predictions)
    ocr_predictions = run_ocr([img_pil], [["en"]], analyzer.det_model, analyzer.det_processor, analyzer.rec_model, analyzer.rec_processor)
    
    layout_result = layout_predictions[0]
    ocr_result = ocr_predictions[0]
    
    ocr_lines = []
    for line in ocr_result.text_lines:
        bx = get_bbox(line.polygon)
        ocr_lines.append({"bbox": bx, "text": line.text})

    def split_into_sub_blocks(x1, y1, x2, y2, block_label):
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
                # Add margin
                # margin = 5
                # px1 = max(x1, x1+cx-margin)
                # py1 = max(y1, y1+cy-margin)
                # px2 = min(x2, x1+cx+cw+margin)
                # py2 = min(y2, y1+cy+ch+margin)
                px1 = x1+cx
                py1 = y1+cy
                px2 = x1+cx+cw
                py2 = y1+cy+ch
                valid_panels.append([px1, py1, px2, py2])
        
        if len(valid_panels) > 1:
            valid_panels.sort(key=lambda b: (b[1] // 50, b[0]))
            return valid_panels
            
        return [[x1, y1, x2, y2]]

    elements = []
    
    for block_idx, block in enumerate(layout_result.bboxes):
        bx1, by1, bx2, by2 = get_bbox(block.polygon)
        block_type = block.label
        
        sub_blocks = split_into_sub_blocks(bx1, by1, bx2, by2, block_type)
        
        for idx, sb in enumerate(sub_blocks):
            sb_x1, sb_y1, sb_x2, sb_y2 = sb
            sb_area = max(1, (sb_x2 - sb_x1) * (sb_y2 - sb_y1))
            
            # Find intersecting text
            intersecting_text = []
            text_area_covered = 0
            
            for line in ocr_lines:
                l_bbox = line["bbox"]
                inter = get_intersection(sb, l_bbox)
                l_area = (l_bbox[2]-l_bbox[0]) * (l_bbox[3]-l_bbox[1])
                
                if l_area > 0 and inter / l_area > 0.4:
                    intersecting_text.append(line["text"])
                    text_area_covered += inter
                    
            text_coverage_ratio = text_area_covered / sb_area
            
            w = sb_x2 - sb_x1
            h = sb_y2 - sb_y1
            
            if block_type in ["Text", "Title", "List", "Caption", "Formula"] or (block_type == "Table" and len(intersecting_text) > 0 and text_coverage_ratio > 0.05):
                raw_text = " ".join(intersecting_text).strip()
                if raw_text:
                    print(f"[{block_type}->TEXT] {raw_text[:50]}... Bounds: {sb}")
            else:
                if w > 50 and h > 50:
                    print(f"[{block_type}->FIGURE] Bounds: {sb}")

if __name__ == '__main__':
    test_hybrid()

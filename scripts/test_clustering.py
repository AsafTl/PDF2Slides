import cv2
import json
import numpy as np
from PIL import Image

from surya.detection import batch_text_detection
from src.analyzer import SlideAnalyzer

def get_intersection(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    return (x_right - x_left) * (y_bottom - y_top)

def test_custom_layout():
    analyzer = SlideAnalyzer(output_dir='tests/craters_staging')
    img_bgr = cv2.imread('tests/craters/page_004.png')
    img_pil = Image.open('tests/craters/page_004.png').convert("RGB")
    
    # Run text detection
    line_predictions = batch_text_detection([img_pil], analyzer.det_model, analyzer.det_processor)[0]
    
    # We also have layout predictions to know roughly where figures are
    from surya.layout import batch_layout_detection
    layout_result = batch_layout_detection([img_pil], analyzer.layout_model, analyzer.layout_processor, [line_predictions])[0]
    
    def get_bbox(polygon):
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

    print("--- RAW TEXT LINES ---")
    lines = []
    for bbox in line_predictions.bboxes:
        lines.append(get_bbox(bbox.polygon))
        
    # Cluster lines that are very close vertically and horizontally
    # A simple greedy approach:
    blocks = []
    for line in lines:
        merged = False
        for block in blocks:
            # Check if line is close to block
            # For text, lines in a paragraph are vertically close and horizontally aligned.
            # Block is [x1, y1, x2, y2]
            bx1, by1, bx2, by2 = block
            lx1, ly1, lx2, ly2 = line
            
            # Vertical gap
            v_gap = max(0, max(by1, ly1) - min(by2, ly2))
            # Horizontal gap
            h_gap = max(0, max(bx1, lx1) - min(bx2, lx2))
            
            # If vertical gap is small (e.g. < 2 * line height) and horizontally overlaps
            line_h = ly2 - ly1
            if v_gap < line_h * 2.5 and h_gap < line_h * 5:
                # Merge
                block[0] = min(bx1, lx1)
                block[1] = min(by1, ly1)
                block[2] = max(bx2, lx2)
                block[3] = max(by2, ly2)
                merged = True
                break
        if not merged:
            blocks.append(line.copy())
            
    # Try merging blocks again to coalesce
    changed = True
    while changed:
        changed = False
        new_blocks = []
        while blocks:
            b1 = blocks.pop(0)
            merged_b1 = False
            for b2 in blocks:
                v_gap = max(0, max(b1[1], b2[1]) - min(b1[3], b2[3]))
                h_gap = max(0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
                line_h = (b1[3] - b1[1]) / 2 # Approx
                if v_gap < line_h * 2.5 and h_gap < line_h * 5:
                    b2[0] = min(b1[0], b2[0])
                    b2[1] = min(b1[1], b2[1])
                    b2[2] = max(b1[2], b2[2])
                    b2[3] = max(b1[3], b2[3])
                    merged_b1 = True
                    changed = True
                    break
            if not merged_b1:
                new_blocks.append(b1)
        blocks = new_blocks

    print(f"Clustered into {len(blocks)} text blocks")
    for b in blocks:
        print(f"  Text Block: {b}")
        
    print("\n--- HANDLING MASSIVE FIGURES ---")
    for block in layout_result.bboxes:
        if block.label in ["Figure", "Picture"]:
            x1, y1, x2, y2 = get_bbox(block.polygon)
            print(f"\nAnalyzing Layout {block.label}: {[x1, y1, x2, y2]}")
            
            # Extract crop
            crop = img_bgr[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            # Threshold using adaptive or Otsu to isolate the ink
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find external contours of the elements inside the figure
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Group contours that are somewhat close to form panels
            # A scatter plot has many points, we want to group them into 1 axis.
            # So apply morphological CLOSE with a massive kernel relative to image size
            panel_kernel_size = max(10, min(x2-x1, y2-y1) // 20) 
            kernel = np.ones((panel_kernel_size, panel_kernel_size), np.uint8)
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            panel_contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_panels = []
            for c in panel_contours:
                cx, cy, cw, ch = cv2.boundingRect(c)
                if cw * ch > ( (x2-x1)*(y2-y1) * 0.05 ): # At least 5% of the massive figure area
                    valid_panels.append([x1+cx, y1+cy, x1+cx+cw, y1+cy+ch])
            
            if len(valid_panels) > 1:
                print(f"  Split into {len(valid_panels)} distinct sub-figures:")
                for vp in valid_panels:
                    print(f"    Sub-figure: {vp}")
            else:
                print(f"  Kept as 1 figure.")

if __name__ == '__main__':
    test_custom_layout()

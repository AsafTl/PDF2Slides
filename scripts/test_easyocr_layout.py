import os
import cv2
import numpy as np
import easyocr

def compute_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def get_grouped_text_blocks(raw_ocr_result, max_vertical_dist=200):
    if not raw_ocr_result or len(raw_ocr_result) == 0:
        return []
        
    bboxes = []
    # easyocr result format: [([[x,y],[x,y],[x,y],[x,y]], 'text', conf), ...]
    for line in raw_ocr_result:
        box = line[0]
        text_str = line[1]
        conf = line[2]
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        bboxes.append({'bbox': [x1, y1, x2, y2], 'text': text_str, 'conf': conf})
        
    # Sort boxes vertically
    bboxes.sort(key=lambda b: b['bbox'][1])
    
    grouped_blocks = []
    
    for box_dict in bboxes:
        if not grouped_blocks:
            grouped_blocks.append(box_dict)
            continue
            
        last_block = grouped_blocks[-1]
        last_bbox = last_block['bbox']
        curr_bbox = box_dict['bbox']
        
        # Check vertical distance
        vertical_dist = curr_bbox[1] - last_bbox[3]
        
        # Check horizontal overlap (they should roughly align if they are the same paragraph)
        horiz_overlap = min(last_bbox[2], curr_bbox[2]) - max(last_bbox[0], curr_bbox[0])
        
        if -50 <= vertical_dist < max_vertical_dist and horiz_overlap > -500:
            # Merge
            new_x1 = min(last_bbox[0], curr_bbox[0])
            new_y1 = min(last_bbox[1], curr_bbox[1])
            new_x2 = max(last_bbox[2], curr_bbox[2])
            new_y2 = max(last_bbox[3], curr_bbox[3])
            
            grouped_blocks[-1]['bbox'] = [new_x1, new_y1, new_x2, new_y2]
            grouped_blocks[-1]['text'] += " " + box_dict['text']
            # average confidence
            grouped_blocks[-1]['conf'] = (grouped_blocks[-1]['conf'] + box_dict['conf']) / 2.0
        else:
            grouped_blocks.append(box_dict)
            
    return grouped_blocks

def filter_nested_boxes(bboxes_list):
    """Removes boxes that are completely inside other boxes"""
    keep = []
    for i, box1 in enumerate(bboxes_list):
        is_nested = False
        for j, box2 in enumerate(bboxes_list):
            if i != j:
                # if box1 is inside box2
                if (box1[0] >= box2[0] - 10 and box1[1] >= box2[1] - 10 and 
                    box1[2] <= box2[2] + 10 and box1[3] <= box2[3] + 10):
                    is_nested = True
                    break
        if not is_nested:
            keep.append(box1)
    return keep

def test_custom_layout(image_path, out_path):
    print("Initializing EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found")
        return
        
    print("Extracting text via OCR...")
    result = reader.readtext(image_path)
    
    text_blocks = get_grouped_text_blocks(result, max_vertical_dist=300)
    print(f"Found {len(text_blocks)} clustered text blocks.")
    
    # --- Figure extraction using edge contours ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Edge detection
    edges = cv2.Canny(gray, 30, 100)
    
    # 2. Mask out text blocks very generously
    for tb in text_blocks:
        x1, y1, x2, y2 = tb['bbox']
        pad = 80
        cv2.rectangle(edges, (max(0, x1-pad), max(0, y1-pad)), (min(img.shape[1], x2+pad), min(img.shape[0], y2+pad)), 0, -1)
        
    # 3. Aggressively dilate remaining edges to merge disjoint figure components 
    kernel = np.ones((150, 150), np.uint8) 
    dilated = cv2.dilate(edges, kernel, iterations=2)
    dilated = cv2.erode(dilated, kernel, iterations=1) # smooth out
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    raw_fig_bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 200 and h > 200:
            raw_fig_bboxes.append([x, y, x + w, y + h])
            
    fig_bboxes = filter_nested_boxes(raw_fig_bboxes)
    print(f"Found {len(fig_bboxes)} figure blocks.")
    
    # Draw visualization
    viz = img.copy()
    for tb in text_blocks:
        x1, y1, x2, y2 = tb['bbox']
        cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 6)
        cv2.putText(viz, f"TEXT: {tb['text'][:20]}...", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
        
    for (x1, y1, x2, y2) in fig_bboxes:
        cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 0, 255), 8)
        cv2.putText(viz, "FIGURE", (x1 + 20, y1 + 60), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8)
        
    cv2.imwrite(out_path, viz)
    print(f"Saved grouped visualization to {out_path}")

if __name__ == "__main__":
    test_custom_layout(r"tests\craters\page_004.png", r"tests\craters\visualized_easyocr_page_004.png")

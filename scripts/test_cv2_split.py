import cv2
import json
import os
import numpy as np

def test_split_figures():
    img_bgr = cv2.imread('tests/craters/page_004.png')
    
    # Let's say Surya gave us the massive bbox for fig_0:
    # Surya Bbox 0: Label=Figure, Bounds=[207, 1036, 5515, 2645]
    # And fig_1 (which is 3 text blocks):
    # Surya Bbox 1: Label=Table, Bounds=[1002, 872, 4714, 1029]
    
    # 1. Test splitting fig_0
    x1, y1, x2, y2 = 207, 1036, 5515, 2645
    crop = img_bgr[y1:y2, x1:x2]
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological closing to group nearby pixels of the same figure
    kernel = np.ones((15, 15), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Fig 0 contours found: {len(contours)}")
    for i, c in enumerate(contours):
        cx, cy, cw, ch = cv2.boundingRect(c)
        if cw * ch > 1000: # Filter noise
            print(f"  sub-figure {i}: {[x1+cx, y1+cy, x1+cx+cw, y1+cy+ch]}")
            
    # 2. Test splitting fig_1
    x1, y1, x2, y2 = 1002, 872, 4714, 1029
    crop = img_bgr[y1:y2, x1:x2]
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((10, 20), np.uint8) # expand horizontally more for text lines
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"\nFig 1 (Table/Text) contours found: {len(contours)}")
    for i, c in enumerate(contours):
        cx, cy, cw, ch = cv2.boundingRect(c)
        if cw * ch > 500: # Filter noise
            print(f"  sub-block {i}: {[x1+cx, y1+cy, x1+cx+cw, y1+cy+ch]}")

if __name__ == '__main__':
    test_split_figures()

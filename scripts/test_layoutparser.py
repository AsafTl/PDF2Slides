import os
import portalocker.utils

original_get_fh = portalocker.utils.Lock._get_fh

def safe_get_fh(self):
    # Sanitize the filename to avoid Windows invalid characters for the lock file
    import pathlib
    original_path = str(self.filename)
    safe_path = original_path.replace("://", "_").replace("|", "_").replace("<", "_").replace(">", "_")
    self.filename = safe_path
    return original_get_fh(self)

portalocker.utils.Lock._get_fh = safe_get_fh

import cv2
import layoutparser as lp

def test_layoutparser(image_path, out_path):
    print("Initializing LayoutParser...")
    model = lp.EfficientDetLayoutModel("lp://PubLayNet/tf_efficientdet_d1")
    
    img = cv2.imread(image_path)
    # layoutparser works in RGB
    img_rgb = img[..., ::-1] 
    
    print("Predicting layout...")
    layout = model.detect(img_rgb)
    
    print(f"Found {len(layout)} elements.")
    for block in layout:
        print(f"Type: {block.type}, Score: {block.score:.2f}, BBox: {block.coordinates}")
        
    print("Drawing visualization...")
    # draw boxes
    viz = lp.draw_box(img_rgb, layout, box_width=3, show_element_type=True)
    # convert back to BGR to save using cv2
    viz_bgr = cv2.cvtColor(np.array(viz), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, viz_bgr)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    import numpy as np
    test_layoutparser(r"tests\craters\page_004.png", r"tests\craters\visualized_lp_page_004.png")

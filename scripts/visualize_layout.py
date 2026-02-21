import cv2
import json
from PIL import Image
import os
import copy
from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection
from surya.postprocessing.heatmap import draw_polys_on_image
from src.analyzer import SlideAnalyzer

def main():
    analyzer = SlideAnalyzer(output_dir='tests/craters_staging')
    img_path = 'tests/craters/page_004.png'
    img_pil = Image.open(img_path).convert("RGB")
    
    line_predictions = batch_text_detection([img_pil], analyzer.det_model, analyzer.det_processor)
    layout_predictions = batch_layout_detection([img_pil], analyzer.layout_model, analyzer.layout_processor, line_predictions)
    
    layout_pred = layout_predictions[0]
    
    # Save the original layout predictions from Surya
    polygons = [p.polygon for p in layout_pred.bboxes]
    labels = [p.label for p in layout_pred.bboxes]
    
    bbox_image = draw_polys_on_image(polygons, copy.deepcopy(img_pil), labels=labels)
    bbox_image.save('tests/craters_staging/page_004_surya_raw.png')
    print("Saved raw surya layout to tests/craters_staging/page_004_surya_raw.png")
    
    # Let's also print out exactly what Surya thinks it is
    for idx, bbox in enumerate(layout_pred.bboxes):
        xs = [p[0] for p in bbox.polygon]
        ys = [p[1] for p in bbox.polygon]
        bounds = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
        print(f"Surya Bbox {idx}: Label={bbox.label}, Bounds={bounds}")
        
if __name__ == '__main__':
    main()

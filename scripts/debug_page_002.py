import cv2
import json
from PIL import Image
from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection
from surya.ocr import run_ocr
from src.analyzer import SlideAnalyzer

analyzer = SlideAnalyzer(output_dir='tests/craters_staging')
img_path = 'tests/craters/page_002.png'
img_bgr = cv2.imread(img_path)
img_pil = Image.open(img_path).convert('RGB')

line_predictions = batch_text_detection([img_pil], analyzer.det_model, analyzer.det_processor)
layout_predictions = batch_layout_detection([img_pil], analyzer.layout_model, analyzer.layout_processor, line_predictions)
ocr_predictions = run_ocr([img_pil], [['en']], analyzer.det_model, analyzer.det_processor, analyzer.rec_model, analyzer.rec_processor)

def get_bbox(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

ocr_lines = []
for line in ocr_predictions[0].text_lines:
    ocr_lines.append({'bbox': get_bbox(line.polygon), 'text': line.text})

for block in layout_predictions[0].bboxes:
    x1, y1, x2, y2 = get_bbox(block.polygon)
    print(f'\n--- Analyzing Block: {block.label} {[x1, y1, x2, y2]} ---')
    
    sub_blocks = analyzer._split_into_sub_blocks(img_bgr, x1, y1, x2, y2, block.label)
    for i, sb in enumerate(sub_blocks):
        print(f'  Sub-block {i}: {sb}')
        for line in ocr_lines:
            l_bbox = line['bbox']
            inter = analyzer._get_intersection(sb, l_bbox)
            l_area = (l_bbox[2]-l_bbox[0]) * (l_bbox[3]-l_bbox[1])
            if l_area > 0 and inter > 0:
                print(f'    Line "{line["text"][:20]}" (Area: {l_area}) Intersect: {inter} Ratio: {inter/l_area:.2f}')

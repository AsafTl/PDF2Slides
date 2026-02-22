"""
Debug page_011.png to trace why inter-figure text is lost.
Shows raw layout blocks, all OCR lines, and which ones survive into elements.
"""
import json
from PIL import Image
from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection
from surya.ocr import run_ocr
from src.analyzer import SlideAnalyzer

PAGE = 'tests/craters/page_011.png'

analyzer = SlideAnalyzer(output_dir='tests/craters_staging')
img_pil = Image.open(PAGE).convert('RGB')
print(f'Image size: {img_pil.size}')

line_predictions = batch_text_detection([img_pil], analyzer.det_model, analyzer.det_processor)
layout_predictions = batch_layout_detection([img_pil], analyzer.layout_model, analyzer.layout_processor, line_predictions)
ocr_predictions = run_ocr([img_pil], [['en']], analyzer.det_model, analyzer.det_processor, analyzer.rec_model, analyzer.rec_processor)

def bbox(polygon):
    xs = [p[0] for p in polygon]; ys = [p[1] for p in polygon]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

print('\n--- SURYA LAYOUT BLOCKS ---')
for b in layout_predictions[0].bboxes:
    x1,y1,x2,y2 = bbox(b.polygon)
    print(f'  [{b.label}] y={y1}-{y2}  x={x1}-{x2}')

print('\n--- ALL OCR LINES ---')
for line in ocr_predictions[0].text_lines:
    x1,y1,x2,y2 = bbox(line.polygon)
    print(f'  y={y1}-{y2}: "{line.text}"')

print('\n--- FINAL ELEMENTS ---')
result = analyzer.analyze_slide(PAGE, slide_index=11)
for el in result['elements']:
    t = el['type']
    if t in ['text','title']:
        print(f'  [{t}] "{el["text"][:80]}"  bbox={el["bbox"][:2]}')
    else:
        print(f'  [fig] bbox={el["bbox"][:2]}')

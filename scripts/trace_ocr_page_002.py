"""
Debug script to trace ALL raw OCR text from Surya on page_002.png
to identify any hallucinated text that doesn't appear on the actual image.
"""
from PIL import Image
from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection
from surya.ocr import run_ocr
from src.analyzer import SlideAnalyzer

analyzer = SlideAnalyzer(output_dir='tests/craters_staging')
img_path = 'tests/craters/page_002.png'
img_pil = Image.open(img_path).convert('RGB')

print(f"Image size: {img_pil.size}")

line_predictions = batch_text_detection([img_pil], analyzer.det_model, analyzer.det_processor)
layout_predictions = batch_layout_detection([img_pil], analyzer.layout_model, analyzer.layout_processor, line_predictions)
ocr_predictions = run_ocr([img_pil], [['en']], analyzer.det_model, analyzer.det_processor, analyzer.rec_model, analyzer.rec_processor)

print("\n--- ALL LAYOUT BLOCKS (raw from Surya) ---")
for b in layout_predictions[0].bboxes:
    xs = [p[0] for p in b.polygon]; ys = [p[1] for p in b.polygon]
    print(f"  [{b.label}] y={int(min(ys))}-{int(max(ys))}")

print("\n--- ALL OCR TEXT LINES (raw from Surya) ---")
for line in ocr_predictions[0].text_lines:
    xs = [p[0] for p in line.polygon]; ys = [p[1] for p in line.polygon]
    print(f"  y={int(min(ys))}-{int(max(ys))}: '{line.text}'")

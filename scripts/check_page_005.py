"""
Spot-check page_005.png for font size issues.
Prints all text elements with their pt sizes for visual inspection.
"""
from src.analyzer import SlideAnalyzer

analyzer = SlideAnalyzer(output_dir='tests/craters_staging')
result = analyzer.analyze_slide('tests/craters/page_005.png', slide_index=5)

texts = [el for el in result['elements'] if el['type'] in ['text', 'title']]
figs  = [el for el in result['elements'] if el['type'] == 'figure']
size_px2pt = lambda px: int(px / (300 / 72.0))

print('=== page_005 — FONT SIZE CHECK ===')
print('Text elements: %d  |  Figures: %d' % (len(texts), len(figs)))
print()
for t in texts:
    x, y, w, h = t['bbox']
    est_px = t['estimated_size']
    pt = size_px2pt(est_px)
    print('  [%s] ~%dpt (px=%d)  y=%d  "%s"' % (t['type'], pt, est_px, y, t['text'][:70]))
print()
for f in figs:
    x, y, w, h = f['bbox']
    print('  [fig] x=%d y=%d w=%d h=%d' % (x, y, w, h))

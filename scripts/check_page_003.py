"""
Spot-check page_003.png for duplication issues.
Prints all text elements showing their text and bbox positions.
"""
from src.analyzer import SlideAnalyzer

analyzer = SlideAnalyzer(output_dir='tests/craters_staging')
result = analyzer.analyze_slide('tests/craters/page_003.png', slide_index=3)

texts = [el for el in result['elements'] if el['type'] in ['text', 'title']]
figs  = [el for el in result['elements'] if el['type'] == 'figure']
size_px2pt = lambda px: int(px / (300 / 72.0))

print('=== page_003 — DUPLICATION CHECK ===')
print('Text elements: %d  |  Figures: %d' % (len(texts), len(figs)))
print()
for t in texts:
    x, y, w, h = t['bbox']
    pt = size_px2pt(t['estimated_size'])
    print('  [%s] ~%dpt  y=%d  "%s"' % (t['type'], pt, y, t['text'][:80]))
print()
for f in figs:
    x, y, w, h = f['bbox']
    print('  [fig] x=%d y=%d w=%d h=%d' % (x, y, w, h))

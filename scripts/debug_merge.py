"""Debug the merge logic to trace which blocks are being merged on page_004."""
import json
from src.analyzer import SlideAnalyzer

analyzer = SlideAnalyzer(output_dir='tests/craters_staging')
result = analyzer.analyze_slide('tests/craters/page_004.png', slide_index=4)

texts = [el for el in result['elements'] if el['type'] in ['text', 'title']]
figs  = [el for el in result['elements'] if el['type'] == 'figure']

# Show full bboxes to see what was merged
print('=== FINAL TEXT BLOCKS ===')
for t in texts:
    x, y, w, h = t['bbox']
    print('  [%s] y=%d..%d  "%s"' % (t['type'], y, y + h, t['text'][:70]))
print()
print('=== FIGURES ===')
for f in figs:
    x, y, w, h = f['bbox']
    print('  [fig] y=%d..%d' % (y, y + h))

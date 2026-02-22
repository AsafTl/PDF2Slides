"""Verify estimated_size values after min(h,w) fix."""
from src.analyzer import SlideAnalyzer

analyzer = SlideAnalyzer(output_dir='tests/craters_staging')

for page, idx in [('tests/craters/page_002.png', 2), ('tests/craters/page_004.png', 4)]:
    result = analyzer.analyze_slide(page, slide_index=idx)
    print('\n=== page_%03d ===' % idx)
    for el in result['elements']:
        t = el['type']
        if t in ['text', 'title']:
            size_px = el['estimated_size']
            # Convert to pt for readability: px / (300/72)
            size_pt = int(size_px / (300 / 72.0))
            print('  [%s] ~%dpt  "%s"' % (t, size_pt, el['text'][:60]))
        else:
            print('  [fig] %s' % el.get('source_file', '')[-30:])

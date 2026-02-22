import json
from src.analyzer import SlideAnalyzer

analyzer = SlideAnalyzer(output_dir='tests/craters_staging')

for page, idx in [('tests/craters/page_002.png', 2), ('tests/craters/page_004.png', 4)]:
    result = analyzer.analyze_slide(page, slide_index=idx)
    texts = [el for el in result['elements'] if el['type'] in ['text', 'title']]
    figs  = [el for el in result['elements'] if el['type'] == 'figure']
    print('\n=== page_%03d ===' % idx)
    print('Text blocks: %d, Figures: %d' % (len(texts), len(figs)))
    for t in texts:
        print('  [%s] %s' % (t['type'], t['text'][:80]))

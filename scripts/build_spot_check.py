"""Build a 2-slide PPTX from page_003 and page_005 for visual spot-checking."""
from src.analyzer import SlideAnalyzer
from src.builder import PPTXBuilder

DPI = 300
STAGING = 'tests/craters_staging'
OUTPUT   = 'tests/spot_check_003_005.pptx'

pages = [
    #('tests/craters/page_003.png', 3),
    ('tests/craters/page_005.png', 5),
]

analyzer = SlideAnalyzer(output_dir=STAGING)
slides_ir = []

for png_path, slide_idx in pages:
    print('Analyzing %s ...' % png_path)
    slide_data = analyzer.analyze_slide(png_path, slide_index=slide_idx)
    #print(slide_data)
    slides_ir.append(slide_data)
    texts = [el for el in slide_data['elements'] if el['type'] in ('text','title')]
    figs  = [el for el in slide_data['elements'] if el['type'] == 'figure']
    print('  -> %d text elements, %d figures' % (len(texts), len(figs)))

ir = {
    'metadata': {'page_count': len(slides_ir), 'dpi': DPI, 'aspect_ratio': 16/9},
    'slides': slides_ir,
}

print('\nBuilding PPTX ...')
builder = PPTXBuilder(ir_data=ir, output_path=OUTPUT, dpi=DPI)
builder.build()
print('Saved to: %s' % OUTPUT)

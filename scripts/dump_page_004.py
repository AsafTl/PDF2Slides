import json
from src.analyzer import SlideAnalyzer
import sys

def main():
    analyzer = SlideAnalyzer(output_dir='tests/craters_staging')
    result = analyzer.analyze_slide('tests/craters/page_004.png', slide_index=4)

    text_blocks = [el for el in result['elements'] if el['type'] in ['text', 'title']]
    figures = [el for el in result['elements'] if el['type'] == 'figure']

    print('--- OUTPUT SUMMARY ---')
    print(f'Total Text Blocks: {len(text_blocks)}')
    print(f'Total Figures: {len(figures)}')
    print(f'Background Color: {result.get("background_color")}')
    print()
    print('--- JSON OUTPUT ---')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()

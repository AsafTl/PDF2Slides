import os
import cv2
import numpy as np
import pytest
from src.analyzer import SlideAnalyzer

@pytest.fixture
def mock_slide_img(tmp_path):
    """
    Creates a simple image with a white background, some text, and a colored square (mock figure).
    """
    img_path = str(tmp_path / "mock_slide.png")
    
    # White background 800x600
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Draw simple text
    cv2.putText(img, 'Header Text', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, 'Body line 1', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    # Draw blue square (figure)
    cv2.rectangle(img, (400, 200), (600, 400), (255, 0, 0), -1)
    
    cv2.imwrite(img_path, img)
    return img_path

def test_extract_background_color(mock_slide_img):
    analyzer = SlideAnalyzer(output_dir="/tmp/dummy")
    
    img = cv2.imread(mock_slide_img)
    # mock layout bounding boxes
    layout = [[400, 200, 600, 400]]
    
    bg_color = analyzer._extract_background_color(img, layout)
    # The image is predominantly white, so median should yield #ffffff
    assert bg_color.lower() == "#ffffff"

def test_analyzer_integration(mock_slide_img, tmp_path):
    analyzer = SlideAnalyzer(output_dir=str(tmp_path))
    slide_ir = analyzer.analyze_slide(mock_slide_img, slide_index=0)
    
    assert slide_ir["slide_index"] == 0
    assert "background_color" in slide_ir
    
    # Verify elements
    elements = slide_ir["elements"]
    assert len(elements) > 0
    
    # We should have text and potentially a figure
    types = [el["type"] for el in elements]
    
    # PPStructure should detect at least one text block or title
    assert "text" in types or "title" in types
    
    # If a figure was detected, verify it was saved
    figures = [el for el in elements if el["type"] == "figure"]
    for fig in figures:
        assert os.path.exists(fig["source_file"])

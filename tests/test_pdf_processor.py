import os
import fitz
import pytest
from src.pdf_processor import PDFProcessor
from PIL import Image

@pytest.fixture
def sample_pdf(tmp_path):
    """
    Creates a temporary, minimal single-page PDF for testing.
    Standard 16:9 aspect ratio in points (e.g., 960x540).
    """
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    # Create a page in 16:9 format
    page = doc.new_page(width=960, height=540)
    # Add some dummy text
    page.insert_text((100, 100), "Test PDF for Rasterization", fontsize=20)
    doc.save(pdf_path)
    doc.close()
    return str(pdf_path)

@pytest.fixture
def staging_dir(tmp_path):
    d = tmp_path / "pdf_staging"
    d.mkdir()
    return str(d)

def test_extract_metadata(sample_pdf, staging_dir):
    processor = PDFProcessor(pdf_path=sample_pdf, output_dir=staging_dir, dpi=300)
    meta = processor.extract_metadata()
    
    assert meta["total_pages"] == 1
    assert meta["base_width_pts"] == 960
    assert meta["base_height_pts"] == 540
    assert meta["aspect_ratio"] == "16:9"

def test_rasterize_pages(sample_pdf, staging_dir):
    processor = PDFProcessor(pdf_path=sample_pdf, output_dir=staging_dir, dpi=300)
    output_files = processor.rasterize_pages()
    
    # Check if a file was created
    assert len(output_files) == 1
    file_path = output_files[0]
    assert os.path.exists(file_path)
    assert file_path.endswith(".png")
    
    # Verify dimensions map to 300 DPI
    # Original width 960 pts -> (960 / 72) * 300 = 4000 pixels
    # Original height 540 pts -> (540 / 72) * 300 = 2250 pixels
    with Image.open(file_path) as img:
        assert img.format == "PNG"
        assert img.width == 4000
        assert img.height == 2250

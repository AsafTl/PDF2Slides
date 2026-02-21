import os
import fitz  # PyMuPDF
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, pdf_path: str, output_dir: str = "/tmp/pdf_staging", dpi: int = 300):
        """
        Initializes the PDF Processor.
        
        Args:
            pdf_path (str): The path to the source PDF file.
            output_dir (str, optional): The directory where rendered PNGs will be temporarily stored. Defaults to "/tmp/pdf_staging".
            dpi (int, optional): The resolution to render the PDF pages. Defaults to 300.
        """
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.dpi = dpi
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def extract_metadata(self) -> Dict:
        """
        Extracts document-level metadata.
        
        Returns:
            Dict: Contains total_pages, and native dimensions.
        """
        try:
            doc = fitz.open(self.pdf_path)
            meta = {
                "total_pages": len(doc),
            }
            if len(doc) > 0:
                page = doc[0]
                rect = page.rect
                # Typically, rect.width and rect.height are in points (1/72 inch).
                meta["base_width_pts"] = rect.width
                meta["base_height_pts"] = rect.height
                
                # Try to determine standard aspect ratios
                ratio = round(rect.width / rect.height, 2)
                if ratio == round(16/9, 2):
                    meta["aspect_ratio"] = "16:9"
                elif ratio == round(4/3, 2):
                    meta["aspect_ratio"] = "4:3"
                else:
                    meta["aspect_ratio"] = "custom"
            
            doc.close()
            return meta
        except Exception as e:
            logger.error(f"Failed to extract metadata from {self.pdf_path}: {e}")
            raise

    def rasterize_pages(self) -> List[str]:
        """
        Renders the PDF pages into high-fidelity raster images (PNGs).
        
        Returns:
            List[str]: A list of absolute paths to the saved PNG files.
        """
        output_files = []
        try:
            doc = fitz.open(self.pdf_path)
            
            # PyMuPDF uses a Matrix for zoom/resolution. 
            # Default is 72 DPI. To get desired DPI, zoom factor is desired_dpi / 72.
            zoom = self.dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                
                file_name = f"page_{page_num + 1:03d}.png"
                file_path = os.path.join(self.output_dir, file_name)
                
                # Save as lossless PNG
                pix.save(file_path)
                logger.info(f"Saved rasterized page to {file_path}")
                output_files.append(file_path)
                
            doc.close()
            return output_files
        except Exception as e:
            logger.error(f"Failed to rasterize pages from {self.pdf_path}: {e}")
            raise

    def process(self) -> Tuple[Dict, List[str]]:
        """
        Convenience method to extract metadata and rasterize in one go.
        """
        meta = self.extract_metadata()
        files = self.rasterize_pages()
        return meta, files

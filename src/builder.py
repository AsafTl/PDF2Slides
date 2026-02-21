import os
import logging
from pptx import Presentation
from pptx.util import Pt
from pptx.dml.color import RGBColor
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PPTXBuilder:
    def __init__(self, ir_data: Dict[str, Any], output_path: str, dpi: int = 300):
        self.ir_data = ir_data
        self.output_path = output_path
        self.dpi = dpi
        self.prs = Presentation()
        
    def px_to_emus(self, px: float) -> int:
        """
        Converts pixel coordinates to PowerPoint English Metric Units (EMUs).
        1 inch = 914,400 EMUs.
        """
        return int(px * (914400 / self.dpi))

    def hex_to_rgb(self, hex_color: str) -> RGBColor:
        """Converts '#RRGGBB' to an RGBColor object."""
        hex_color = hex_color.lstrip('#')
        # Handle fallback for unrecognized format
        if len(hex_color) != 6:
            return RGBColor(255, 255, 255)
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return RGBColor(r, g, b)

    def build(self) -> str:
        """
        Constructs the .pptx file from the IR dictionary and saves it.
        Returns the saved file path.
        """
        meta = self.ir_data.get("presentation_meta", {})
        
        # Set Master Canvas Dimensions
        base_width_px = meta.get("base_width_px", 1920)
        base_height_px = meta.get("base_height_px", 1080)
        
        self.prs.slide_width = self.px_to_emus(base_width_px)
        self.prs.slide_height = self.px_to_emus(base_height_px)
        
        blank_slide_layout = self.prs.slide_layouts[6] # Typically blank
        
        slides = self.ir_data.get("slides", [])
        
        for slide_data in slides:
            logger.info(f"Building Slide {slide_data.get('slide_index', 0)}...")
            slide = self.prs.slides.add_slide(blank_slide_layout)
            
            # Apply Background solid color
            bg_hex = slide_data.get("background_color", "#FFFFFF")
            background = slide.background
            fill = background.fill
            fill.solid()
            fill.fore_color.rgb = self.hex_to_rgb(bg_hex)
            
            elements = slide_data.get("elements", [])
            for element in elements:
                el_type = element.get("type")
                bbox = element.get("bbox") # [x, y, w, h]
                
                if not bbox or len(bbox) != 4:
                    continue
                    
                x, y, w, h = bbox
                left = self.px_to_emus(x)
                top = self.px_to_emus(y)
                width = self.px_to_emus(w)
                height = self.px_to_emus(h)
                
                if el_type == "figure":
                    source_file = element.get("source_file")
                    if source_file and os.path.exists(source_file):
                        try:
                            slide.shapes.add_picture(source_file, left, top, width, height)
                        except Exception as e:
                            logger.error(f"Failed to add picture {source_file}: {e}")
                            
                elif el_type in ["title", "text"]:
                    text_str = element.get("text", "")
                    if not text_str:
                        continue
                        
                    textbox = slide.shapes.add_textbox(left, top, width, height)
                    text_frame = textbox.text_frame
                    text_frame.word_wrap = True
                    
                    p = text_frame.paragraphs[0]
                    p.text = text_str
                    
                    # Formatting Estimation
                    estimated_size_px = element.get("estimated_size", 24)
                    
                    # Convert assumed font pixel heights into Points (Pt)
                    # For a 300 DPI image, 1 pt = ~4.16 pixels (300/72)
                    pt_size = max(8, int(estimated_size_px / (self.dpi / 72.0)))
                    
                    for run in p.runs:
                        run.font.size = Pt(pt_size)
                        run.font.bold = element.get("is_bold", False)
                        run.font.name = 'Arial' # Standard fallback
                        run.font.color.rgb = RGBColor(0, 0, 0) # Assumed default for legibility

        logger.info(f"Saving final presentation to {self.output_path}")
        self.prs.save(self.output_path)
        return self.output_path

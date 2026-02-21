import os
import json
import shutil
import logging
import argparse
from typing import Optional

from src.pdf_processor import PDFProcessor
from src.analyzer import SlideAnalyzer
from src.builder import PPTXBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DPI = 300


def run_pipeline(pdf_path: str, output_path: Optional[str] = None, staging_dir: Optional[str] = None, keep_staging: bool = False) -> str:
    """
    Runs the full PDF → PPTX pipeline.

    Args:
        pdf_path:     Path to the source PDF file.
        output_path:  Path for the output .pptx file. Defaults to same dir as PDF.
        staging_dir:  Temporary directory for PNGs and figure crops. Defaults to <pdf_dir>/staging.
        keep_staging: If True, do not delete the staging directory after completion.

    Returns:
        The path of the generated .pptx file.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pdf_dir = os.path.dirname(os.path.abspath(pdf_path))
    pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]

    if staging_dir is None:
        staging_dir = os.path.join(pdf_dir, f"{pdf_stem}_staging")

    if output_path is None:
        output_path = os.path.join(pdf_dir, f"{pdf_stem}.pptx")

    os.makedirs(staging_dir, exist_ok=True)

    # ── Phase 1: Rasterize PDF ──────────────────────────────────────────────
    logger.info("=== Phase 1: Rasterizing PDF ===")
    processor = PDFProcessor(pdf_path=pdf_path, output_dir=staging_dir, dpi=DPI)
    meta, png_paths = processor.process()
    logger.info(f"Rasterized {len(png_paths)} pages.")

    # Build presentation_meta for IR
    # Use actual pixel dimensions of the first rendered PNG
    import cv2
    first_img = cv2.imread(png_paths[0])
    if first_img is None:
        raise RuntimeError(f"Could not read first rasterized page: {png_paths[0]}")
    base_height_px, base_width_px = first_img.shape[:2]

    presentation_meta = {
        "total_slides": len(png_paths),
        "aspect_ratio": meta.get("aspect_ratio", "custom"),
        "base_width_px": base_width_px,
        "base_height_px": base_height_px,
        "dpi": DPI,
    }

    # ── Phase 2: Analyze Slides ─────────────────────────────────────────────
    logger.info("=== Phase 2: Analyzing Slides ===")
    analyzer = SlideAnalyzer(output_dir=staging_dir)

    slides_ir = []
    for slide_idx, png_path in enumerate(png_paths):
        logger.info(f"  Analyzing slide {slide_idx + 1}/{len(png_paths)}: {os.path.basename(png_path)}")
        slide_data = analyzer.analyze_slide(png_path, slide_index=slide_idx)
        slides_ir.append(slide_data)

    ir = {
        "presentation_meta": presentation_meta,
        "slides": slides_ir,
    }

    # Save IR JSON alongside the output for inspection / debugging
    ir_path = os.path.join(pdf_dir, f"{pdf_stem}_ir.json")
    with open(ir_path, "w", encoding="utf-8") as f:
        json.dump(ir, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved Intermediate Representation to {ir_path}")

    # ── Phase 3: Build PPTX ─────────────────────────────────────────────────
    logger.info("=== Phase 3: Building PPTX ===")
    builder = PPTXBuilder(ir_data=ir, output_path=output_path, dpi=DPI)
    builder.build()
    logger.info(f"Presentation saved to: {output_path}")

    # ── Cleanup ─────────────────────────────────────────────────────────────
    if not keep_staging:
        logger.info(f"Cleaning up staging directory: {staging_dir}")
        shutil.rmtree(staging_dir, ignore_errors=True)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert a PDF presentation to PPTX.")
    parser.add_argument("pdf_path", help="Path to the source PDF file.")
    parser.add_argument("--output", "-o", default=None, help="Output .pptx path (default: same dir as PDF).")
    parser.add_argument("--staging", default=None, help="Staging directory for PNGs (default: <pdf>_staging/).")
    parser.add_argument("--keep-staging", action="store_true", help="Do not delete the staging directory on completion.")
    args = parser.parse_args()

    result = run_pipeline(
        pdf_path=args.pdf_path,
        output_path=args.output,
        staging_dir=args.staging,
        keep_staging=args.keep_staging,
    )
    print(f"\n[DONE] Output saved to: {result}")


if __name__ == "__main__":
    main()

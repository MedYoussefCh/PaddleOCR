"""Batch PDF-to-text extraction with PaddleOCR-VL (offline-friendly).

This script renders each PDF page to an image, runs PaddleOCR-VL, and saves the
Markdown output as a per-page `.txt` file using the naming convention
`<pdf_filename>.pageXX.txt`.
"""
from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path
from typing import Iterable, Tuple

import pypdfium2 as pdfium
from paddleocr import PaddleOCRVL


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse every PDF in a folder with PaddleOCR-VL and save page-wise text files.",
    )
    parser.add_argument(
        "--input_dir",
        default="input",
        help="Directory containing PDF files to parse (default: input).",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        help="Directory where per-page .txt files will be written (default: output).",
    )
    parser.add_argument(
        "--layout_detection_model_dir",
        required=True,
        help="Local directory for the layout detection model (e.g., PP-DocLayout_plus-L).",
    )
    parser.add_argument(
        "--vl_rec_model_dir",
        required=True,
        help="Local directory for the VLM recognition model (e.g., PaddleOCR-VL-0.9B).",
    )
    parser.add_argument(
        "--vl_rec_model_name",
        default="PaddleOCR-VL-0.9B",
        help="Logical name of the VLM model (default: PaddleOCR-VL-0.9B).",
    )
    parser.add_argument(
        "--doc_orientation_classify_model_dir",
        help="Local directory for the optional doc orientation classifier (if enabled).",
    )
    parser.add_argument(
        "--doc_unwarping_model_dir",
        help="Local directory for the optional doc unwarping model (if enabled).",
    )
    parser.add_argument(
        "--use_doc_orientation_classify",
        action="store_true",
        help="Enable document orientation classification (requires the corresponding model).",
    )
    parser.add_argument(
        "--use_doc_unwarping",
        action="store_true",
        help="Enable document unwarping (requires the corresponding model).",
    )
    parser.add_argument(
        "--use_chart_recognition",
        action="store_true",
        help="Enable chart recognition (requires the chart recognition model).",
    )
    parser.add_argument(
        "--render_scale",
        type=float,
        default=1.0,
        help=(
            "Scale factor applied when rendering PDF pages to images. Increase slightly "
            "(e.g., 1.25–1.5) for small text, decrease to save memory."
        ),
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        help="Optional minimum pixel count passed through to the pipeline (see PaddleOCR-VL docs).",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        help="Optional maximum pixel count passed through to the pipeline (see PaddleOCR-VL docs).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for troubleshooting.",
    )
    return parser.parse_args()


def render_pdf_pages(pdf_path: Path) -> Iterable[Tuple[int, int, "pdfium.PdfPage"]]:
    """Yield (page_index, total_pages, page) for each page in the PDF."""
    doc = pdfium.PdfDocument(str(pdf_path))
    total_pages = len(doc)
    for page_index in range(total_pages):
        yield page_index, total_pages, doc[page_index]


def save_page_result_to_txt(
    md_folder: Path, image_stem: str, output_dir: Path, pdf_name: str, page_index: int, total_pages: int
) -> Path:
    md_path = md_folder / f"{image_stem}.md"
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown output not found at {md_path}")

    digits = max(2, len(str(total_pages)))
    page_suffix = f"page{page_index:0{digits}d}"
    output_path = output_dir / f"{pdf_name}.{page_suffix}.txt"
    output_path.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading PaddleOCR-VL with local models…")
    ocr = PaddleOCRVL(
        layout_detection_model_dir=args.layout_detection_model_dir,
        vl_rec_model_name=args.vl_rec_model_name,
        vl_rec_model_dir=args.vl_rec_model_dir,
        doc_orientation_classify_model_dir=args.doc_orientation_classify_model_dir,
        doc_unwarping_model_dir=args.doc_unwarping_model_dir,
        use_doc_orientation_classify=args.use_doc_orientation_classify,
        use_doc_unwarping=args.use_doc_unwarping,
        use_chart_recognition=args.use_chart_recognition,
    )

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", input_dir)
        return

    for pdf_path in pdf_files:
        logger.info("Processing %s", pdf_path.name)
        with tempfile.TemporaryDirectory(prefix="paddleocr_vl_pdf_") as work_dir:
            work_dir_path = Path(work_dir)
            for idx, total, page in render_pdf_pages(pdf_path):
                pil_image = page.render(scale=args.render_scale).to_pil()
                image_stem = f"{pdf_path.name}.page{idx + 1:02d}"
                page_image_path = work_dir_path / f"{image_stem}.png"
                pil_image.save(page_image_path, format="PNG")

                results = ocr.predict(
                    str(page_image_path),
                    min_pixels=args.min_pixels,
                    max_pixels=args.max_pixels,
                )
                if not results:
                    logger.warning("No OCR results for %s (page %s)", pdf_path.name, idx + 1)
                    continue

                # Save the pipeline's markdown output, then copy it to the desired .txt file name.
                results[0].save_to_markdown(save_path=str(work_dir_path))
                output_path = save_page_result_to_txt(
                    work_dir_path,
                    page_image_path.stem,
                    output_dir,
                    pdf_path.name,
                    idx + 1,
                    total,
                )
                logger.info("Saved %s", output_path)


if __name__ == "__main__":
    main()

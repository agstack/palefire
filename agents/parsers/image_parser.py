"""
Pale Fire - Image Parser

Extract text and metadata from image files using Tesseract OCR.
Supports JPG, JPEG, PNG, WEBP, TIFF, BMP.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import os
import subprocess
import logging

from .base_parser import BaseParser, ParseResult

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False


class ImageParser(BaseParser):
    """Parser for image files using Tesseract OCR."""

    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.tiff', '.bmp']

    def get_supported_extensions(self) -> List[str]:
        return self.supported_extensions

    def parse(self, file_path: str, **kwargs) -> ParseResult:
        """
        Parse image file and extract text via Tesseract OCR.

        Args:
            file_path: Path to image file
            lang: OCR language (default 'spa+eng')
            psm: Page segmentation mode (default 3)

        Returns:
            ParseResult with extracted text and image metadata
        """
        if not self.validate_file(file_path):
            return ParseResult(
                text="",
                metadata={},
                error=f"File not found or invalid: {file_path}"
            )

        lang = kwargs.get('lang', 'spa+eng')
        psm = kwargs.get('psm', 3)
        extracted_text = ""
        width = 0
        height = 0
        mode = ""
        format_name = ""

        # Try PIL + pytesseract first if available
        if PYTESSERACT_AVAILABLE:
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    mode = img.mode
                    format_name = img.format or Path(file_path).suffix.lstrip('.').upper()
                    
                    config = f"--psm {psm}"
                    try:
                        extracted_text = pytesseract.image_to_string(img, lang=lang, config=config).strip()
                    except Exception:
                        extracted_text = pytesseract.image_to_string(img, config=config).strip()
                        
                    # If empty, try fallback PSM modes (6 or 11)
                    if not extracted_text:
                        for fallback_psm in [6, 11]:
                            try:
                                t = pytesseract.image_to_string(img, lang=lang, config=f"--psm {fallback_psm}").strip()
                                if t:
                                    extracted_text = t
                                    break
                            except Exception:
                                pass
            except Exception as e:
                logger.warning(f"PIL/pytesseract failed on {file_path}: {e}")

        # Fallback to direct tesseract CLI if text is still empty
        if not extracted_text:
            for opt_lang in [lang, "spa", "eng"]:
                try:
                    res = subprocess.run(
                        ["tesseract", file_path, "stdout", "-l", opt_lang, "--psm", str(psm)],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if res.returncode == 0 and res.stdout.strip():
                        extracted_text = res.stdout.strip()
                        break
                except Exception as e:
                    logger.debug(f"tesseract CLI failed with lang {opt_lang}: {e}")

        # If still empty, provide basic visual description fallback
        if not extracted_text:
            extracted_text = f"[Image file {Path(file_path).name}: {width}x{height} pixels, format {format_name or Path(file_path).suffix.upper()}]"

        metadata = {
            'file_name': Path(file_path).name,
            'file_path': str(Path(file_path).resolve()),
            'file_size': os.path.getsize(file_path),
            'file_type': 'image',
            'format': format_name,
            'width': width,
            'height': height,
            'color_mode': mode,
            'ocr_engine': 'tesseract',
            'ocr_language': lang
        }

        return ParseResult(
            text=extracted_text,
            metadata=metadata
        )

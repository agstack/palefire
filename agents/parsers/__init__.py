"""
Pale Fire - File Parsers

Parsers for extracting text from various file formats.
"""

from .base_parser import BaseParser, ParseResult
from .txt_parser import TXTParser
from .csv_parser import CSVParser
from .pdf_parser import PDFParser
from .spreadsheet_parser import SpreadsheetParser
from .url_parser import URLParser
from .image_parser import ImageParser

__all__ = [
    'BaseParser',
    'ParseResult',
    'TXTParser',
    'CSVParser',
    'PDFParser',
    'SpreadsheetParser',
    'URLParser',
    'ImageParser',
]

# Parser registry
PARSERS = {
    '.txt': TXTParser,
    '.csv': CSVParser,
    '.pdf': PDFParser,
    '.xlsx': SpreadsheetParser,
    '.xls': SpreadsheetParser,
    '.ods': SpreadsheetParser,
    '.jpg': ImageParser,
    '.jpeg': ImageParser,
    '.png': ImageParser,
    '.webp': ImageParser,
    '.tiff': ImageParser,
    '.bmp': ImageParser,
}


def is_url(path: str) -> bool:
    """
    Check if the given path is a URL.
    
    Args:
        path: Path or URL to check
        
    Returns:
        True if path is a URL, False otherwise
    """
    from urllib.parse import urlparse
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def get_parser(file_path: str) -> BaseParser:
    """
    Get appropriate parser for file based on extension or URL.
    
    Args:
        file_path: Path to file or URL
        
    Returns:
        Parser instance
        
    Raises:
        ValueError: If no parser available for file type
    """
    # Check if it's a URL
    if is_url(file_path):
        return URLParser()
    
    from pathlib import Path
    ext = Path(file_path).suffix.lower()
    
    if ext not in PARSERS:
        raise ValueError(f"No parser available for file type: {ext}")
    
    return PARSERS[ext]()


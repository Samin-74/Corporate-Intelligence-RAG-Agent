"""
PDF Ingestion Pipeline
- Primary: Docling for advanced PDF->Markdown conversion (tables, layouts)
- Fallback: PyMuPDF for basic text extraction
- Splits text into semantic chunks with overlap
- Preserves page-level metadata for source attribution
"""

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass
class DocumentChunk:
    """A chunk of text with metadata for traceability."""
    text: str
    source_file: str
    page_number: int
    chunk_index: int

    def to_metadata(self) -> dict:
        return {
            "source_file": self.source_file,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
        }


def extract_text_from_pdf(pdf_path: str | Path) -> list[dict]:
    """
    Extract text from a PDF file using Docling (advanced) or PyMuPDF (fallback).
    
    Docling converts PDFs to Markdown, preserving:
    - Tables (converted to Markdown tables)
    - Headings and structure
    - Lists and formatting
    - Layout and reading order
    
    Returns:
        List of dicts with keys: 'text', 'page_number', 'source_file'
    """
    pdf_path = Path(pdf_path)
    
    # Try Docling first (best for financial documents with tables)
    try:
        from docling.document_converter import DocumentConverter
        
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        
        # Export to Markdown
        markdown_text = result.document.export_to_markdown()
        
        # Docling doesn't give us page-level breakdown easily, 
        # so we'll treat the whole doc as one page or split by sections
        pages = []
        
        # Try to split by page markers if present, otherwise chunk by headings
        page_sections = split_markdown_by_pages(markdown_text)
        
        for page_num, section_text in enumerate(page_sections, 1):
            if section_text.strip():
                pages.append({
                    "text": section_text.strip(),
                    "page_number": page_num,
                    "source_file": pdf_path.name,
                })
        
        if pages:
            return pages
            
    except ImportError:
        # Docling not installed, will use fallback
        pass
    except Exception as e:
        # Docling failed, will use fallback
        print(f"Docling extraction failed: {e}, using fallback...")
    
    # Fallback to PyMuPDF
    return extract_text_with_pymupdf(pdf_path)


def split_markdown_by_pages(markdown: str) -> list[str]:
    """
    Split markdown into logical page-like sections.
    Since Docling doesn't preserve exact page numbers, we split by major headings.
    """
    # Split on major headings (## or ###) to create logical sections
    sections = re.split(r'\n(?=#+\s)', markdown)
    
    # If no headings found, split into ~2000 char chunks (approx 1 page)
    if len(sections) <= 1:
        chunk_size = 2000
        sections = [markdown[i:i+chunk_size] for i in range(0, len(markdown), chunk_size)]
    
    return [s.strip() for s in sections if s.strip()]


def extract_text_with_pymupdf(pdf_path: Path) -> list[dict]:
    """
    Fallback extraction using PyMuPDF.
    """
    import fitz
    
    pages = []
    doc = fitz.open(str(pdf_path))
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Use blocks extraction for best layout preservation
        blocks = page.get_text("blocks")
        block_text = "\n".join([block[4] for block in blocks if len(block) > 4])
        
        if not block_text.strip():
            # Fallback to plain text
            block_text = page.get_text("text")
        
        if block_text.strip():
            pages.append({
                "text": block_text.strip(),
                "page_number": page_num + 1,
                "source_file": pdf_path.name,
            })
    
    doc.close()
    return pages


def clean_text_preserve_structure(text: str) -> str:
    """
    Clean text while preserving important structure like tables.
    
    - Removes excessive blank lines (more than 2 consecutive)
    - Preserves single/double line breaks (important for tables)
    - Removes trailing whitespace on lines
    - Preserves indentation that might indicate structure
    """
    # Remove trailing whitespace from each line
    lines = text.split("\n")
    cleaned_lines = [line.rstrip() for line in lines]
    
    # Reduce excessive consecutive blank lines (keep max 2)
    result_lines = []
    blank_count = 0
    for line in cleaned_lines:
        if not line.strip():
            blank_count += 1
            if blank_count <= 2:
                result_lines.append(line)
        else:
            blank_count = 0
            result_lines.append(line)
    
    return "\n".join(result_lines)


def split_text_into_chunks(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[str]:
    """
    Split text into overlapping chunks with intelligent boundary detection.
    Designed to handle financial documents with tables and structured data.
    
    Strategy:
    1. Try to split at semantic boundaries (paragraphs, sections)
    2. Preserve table rows together when possible
    3. Respect sentence boundaries
    4. Use overlap to maintain context across chunks

    Args:
        text: The input text to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of text chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    # Separators in order of preference (preserve semantic boundaries)
    # For financial docs: preserve section breaks, tables, and sentence structure
    separators = [
        "\n\n\n",      # Section breaks
        "\n\n",        # Paragraph breaks
        "\n",          # Line breaks (critical for tables)
        ". ",          # Sentence ends
        "! ",
        "? ",
        "; ",
        ", ",
        " ",           # Last resort: word boundaries
    ]

    chunks = []
    current_pos = 0

    while current_pos < len(text):
        # Determine the end of this chunk
        end_pos = min(current_pos + chunk_size, len(text))

        # If we're not at the end, try to find a good split point
        if end_pos < len(text):
            split_pos = None
            
            # Try each separator in order of preference
            for sep in separators:
                # Look backwards from end_pos for the separator
                search_text = text[current_pos:end_pos]
                last_sep = search_text.rfind(sep)
                
                # Only accept splits that are past 30% of chunk_size
                # This prevents tiny chunks while still respecting boundaries
                if last_sep != -1 and last_sep > chunk_size * 0.3:
                    split_pos = current_pos + last_sep + len(sep)
                    break

            if split_pos is not None:
                end_pos = split_pos

        # Extract chunk and clean it
        chunk_text = text[current_pos:end_pos].strip()
        
        # Only add non-empty chunks
        if chunk_text:
            chunks.append(chunk_text)

        # Move forward, accounting for overlap
        next_pos = end_pos - chunk_overlap

        # Safety: always advance by at least 1 character to prevent infinite loop
        if next_pos <= current_pos:
            next_pos = end_pos

        current_pos = next_pos

    return chunks


def process_pdf(
    pdf_path: str | Path,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[DocumentChunk]:
    """
    Full ingestion pipeline: PDF → pages → chunks with metadata.

    Args:
        pdf_path: Path to the PDF file.
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of DocumentChunk objects ready for embedding.
    """
    pages = extract_text_from_pdf(pdf_path)
    
    if not pages:
        raise ValueError(f"No text content extracted from {pdf_path}")
    
    all_chunks = []
    chunk_idx = 0

    for page in pages:
        text_chunks = split_text_into_chunks(
            page["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for text in text_chunks:
            # Skip empty chunks
            if text.strip():
                all_chunks.append(DocumentChunk(
                    text=text,
                    source_file=page["source_file"],
                    page_number=page["page_number"],
                    chunk_index=chunk_idx,
                ))
                chunk_idx += 1

    if not all_chunks:
        raise ValueError(f"No valid chunks created from {pdf_path}")

    return all_chunks


def process_multiple_pdfs(
    pdf_paths: list[str | Path],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[DocumentChunk]:
    """Process multiple PDFs and return all chunks."""
    all_chunks = []
    for pdf_path in pdf_paths:
        chunks = process_pdf(pdf_path, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
    return all_chunks

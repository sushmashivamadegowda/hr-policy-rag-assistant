import os
import io
import re
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import pypdf
import pdfplumber
import docx2txt

def decode_text_bytes(file_bytes: bytes) -> str:
    """Attempts decoding text with common encodings gracefully."""
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            return file_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="ignore")

def parse_pdf(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    Extract text and tables page by page from PDF files using pdfplumber,
    falling back to pypdf if needed. Formats tables cleanly as markdown.
    """
    documents = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for idx, page in enumerate(pdf.pages):
                page_number = idx + 1
                page_text_parts = []
                
                # 1. Extract tables as structured markdown text
                try:
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            clean_table = [
                                [str(cell).strip() if cell is not None else "" for cell in row]
                                for row in table if any(row)
                            ]
                            if clean_table and len(clean_table) > 1:
                                headers = clean_table[0]
                                header_row = "| " + " | ".join(headers) + " |"
                                sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
                                data_rows = [
                                    "| " + " | ".join(r) + " |" for r in clean_table[1:]
                                ]
                                page_text_parts.append("\n".join([header_row, sep_row] + data_rows))
                except Exception as tbl_err:
                    print(f"Table extraction notice on page {page_number} of {filename}: {tbl_err}")
                
                # 2. Extract prose text
                text = page.extract_text(layout=False) or ""
                text = text.strip()
                if text:
                    page_text_parts.append(text)
                    
                combined = "\n\n".join(page_text_parts).strip()
                if combined:
                    documents.append({
                        "text": combined,
                        "source": filename,
                        "page": page_number
                    })
    except Exception as e:
        print(f"pdfplumber notice for {filename} ({e}), falling back to pypdf...")
        try:
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            for idx, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                if text:
                    documents.append({
                        "text": text,
                        "source": filename,
                        "page": idx + 1
                    })
        except Exception as fallback_e:
            print(f"pypdf fallback also failed for {filename}: {fallback_e}")
            
    return documents

def parse_docx(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    Extract text from DOCX files.
    Safely validates ZIP format to prevent crashing on legacy binary .doc files.
    """
    # Check for legacy binary .doc (OLE2 magic number: D0 CF 11 E0 A1 B1 1A E1)
    if file_bytes.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'):
        print(f"Notice: '{filename}' is a legacy Word 97-2003 (.doc) binary file. "
              f"Attempting text stream recovery. Consider saving as .docx for complete fidelity.")
        raw_strings = re.findall(rb'[\x20-\x7E\r\n]{4,}', file_bytes)
        recovered = " ".join([s.decode('latin-1', errors='ignore') for s in raw_strings])
        recovered = re.sub(r'\s+', ' ', recovered).strip()
        if recovered:
            return [{"text": recovered, "source": filename, "page": 1}]
        return []

    # Check if it's a valid zip file (standard for .docx: PK\x03\x04)
    if not file_bytes.startswith(b'PK\x03\x04'):
        print(f"Error: '{filename}' is not a valid docx archive format.")
        return []

    try:
        text = docx2txt.process(io.BytesIO(file_bytes)) or ""
        text = text.strip()
        if text:
            return [{
                "text": text,
                "source": filename,
                "page": 1
            }]
    except Exception as e:
        print(f"Error extracting DOCX text from {filename}: {e}")
        
    return []

def parse_html(file_content: str, filename: str) -> List[Dict[str, Any]]:
    """Parse HTML content extracting clean body text and heading hierarchy."""
    soup = BeautifulSoup(file_content, "html.parser")
    
    # Remove script, style, header, footer, navigation elements
    for element in soup(["script", "style", "header", "footer", "nav", "noscript"]):
        element.decompose()
        
    text = soup.get_text(separator="\n")
    # Clean blank lines while preserving meaningful breaks
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    cleaned_text = "\n".join(chunk for chunk in chunks if chunk)
    
    return [{
        "text": cleaned_text,
        "source": filename,
        "page": 1
    }]

def parse_txt(file_content: str, filename: str) -> List[Dict[str, Any]]:
    """Parse raw plain text, Markdown, CSV or JSON files."""
    cleaned_text = file_content.strip()
    if not cleaned_text:
        return []
    return [{
        "text": cleaned_text,
        "source": filename,
        "page": 1
    }]

def extract_documents_from_file(file_name: str, file_bytes: bytes) -> List[Dict[str, Any]]:
    """Generic entry point for parsing various file formats."""
    ext = os.path.splitext(file_name)[1].lower()
    
    try:
        if ext == ".pdf":
            return parse_pdf(file_bytes, file_name)
        elif ext in [".docx", ".doc"]:
            return parse_docx(file_bytes, file_name)
        elif ext in [".html", ".htm"]:
            text_content = decode_text_bytes(file_bytes)
            return parse_html(text_content, file_name)
        elif ext in [".txt", ".md", ".csv", ".json"]:
            text_content = decode_text_bytes(file_bytes)
            return parse_txt(text_content, file_name)
        else:
            # Fallback text decoder
            text_content = decode_text_bytes(file_bytes)
            return parse_txt(text_content, file_name)
    except Exception as e:
        print(f"Error parsing file {file_name}: {e}")
        return []

def load_documents_from_directory(dir_path: str) -> List[Dict[str, Any]]:
    """Scans directory and extracts documents from all supported files."""
    all_documents = []
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        return all_documents
        
    for root, _, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
                docs = extract_documents_from_file(file, file_bytes)
                all_documents.extend(docs)
            except Exception as e:
                print(f"Failed loading file {file}: {e}")
                
    return all_documents

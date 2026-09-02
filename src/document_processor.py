import io
import os
from typing import List, Dict, Any, Tuple
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """Handles parsing, text extraction, page tracking, and chunking of documents."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_pdf_bytes(self, file_bytes: bytes, filename: str) -> List[Document]:
        """Extract text page by page from PDF bytes."""
        documents = []
        pdf_reader = PdfReader(io.BytesIO(file_bytes))
        total_pages = len(pdf_reader.pages)
        
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                metadata = {
                    "source": filename,
                    "page": i + 1,
                    "total_pages": total_pages,
                    "file_type": "pdf"
                }
                documents.append(Document(page_content=text, metadata=metadata))
        return documents

    def load_text_bytes(self, file_bytes: bytes, filename: str) -> List[Document]:
        """Extract text from plain text or Markdown bytes."""
        text = file_bytes.decode("utf-8", errors="ignore")
        metadata = {
            "source": filename,
            "page": 1,
            "total_pages": 1,
            "file_type": "txt"
        }
        return [Document(page_content=text, metadata=metadata)]

    def process_file(self, file_bytes: bytes, filename: str) -> Tuple[List[Document], List[Document]]:
        """
        Process a file, return (raw_pages, chunked_documents).
        Assigns unique chunk_id to each chunk metadata.
        """
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".pdf":
            raw_docs = self.load_pdf_bytes(file_bytes, filename)
        else:
            raw_docs = self.load_text_bytes(file_bytes, filename)

        if not raw_docs:
            return [], []

        chunks = self.text_splitter.split_documents(raw_docs)
        
        # Attach unique chunk index and token length metadata
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"{filename}_chunk_{idx+1}"
            chunk.metadata["chunk_index"] = idx + 1
            chunk.metadata["char_count"] = len(chunk.page_content)
            chunk.metadata["word_count"] = len(chunk.page_content.split())

        return raw_docs, chunks

    @staticmethod
    def get_document_stats(chunks: List[Document]) -> Dict[str, Any]:
        """Compute statistical breakdown of the chunked documents."""
        if not chunks:
            return {
                "total_chunks": 0,
                "total_words": 0,
                "avg_chunk_length": 0,
                "min_chunk_length": 0,
                "max_chunk_length": 0,
                "sources": []
            }
        
        char_lens = [c.metadata.get("char_count", len(c.page_content)) for c in chunks]
        words = [c.metadata.get("word_count", len(c.page_content.split())) for c in chunks]
        sources = sorted(list(set(c.metadata.get("source", "Unknown") for c in chunks)))

        return {
            "total_chunks": len(chunks),
            "total_words": sum(words),
            "avg_chunk_length": round(sum(char_lens) / len(char_lens), 1),
            "min_chunk_length": min(char_lens),
            "max_chunk_length": max(char_lens),
            "sources": sources
        }

import datetime
import re
import hashlib
import logging
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.api.models.Collection import Collection
from pathlib import Path
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Language(Enum):
    ENGLISH = "en"
    VIETNAMESE = "vi"
    UNKNOWN = "unknown"


class PDFProcessor:
    def __init__(
            self,
            chunk_size: int = 400,
            chunk_overlap: int = 50,
            max_workers: int = 4,
            min_chunk_length: int = 50
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_workers = max_workers
        self.min_chunk_length = min_chunk_length
        self.separator_patterns = {
            Language.ENGLISH: ["\n\n# ", "\n\n## ", "\n\n### ", "\n\nChapter ", "\n\n", "\n", ". ", " ", ""],
            Language.VIETNAMESE: ["\n\n# ", "\n\n## ", "\n\nChương ", "\n\nPhần ", "\n\n", "\n", ". ", " ", ""],
            Language.UNKNOWN: ["\n\n", "\n", ". ", " ", ""]
        }

    def clean_text(self, text: str) -> str:
        if not text: return ""
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'\xa0', ' ', text)
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'([.,!?])(\S)', r'\1 \2', text)
        text = re.sub(r'-{2,}', '—', text)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        return text.strip()

    def detect_language(self, text: str) -> Language:
        if not text: return Language.UNKNOWN
        sample = text.lower()
        vietnamese_chars = set('ăâđêôơưàáạảãầấậẩẫằắặẳẵèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ')
        if any(char in vietnamese_chars for char in sample[:500]):
            return Language.VIETNAMESE
        english_indicators = ['the', 'and', 'that', 'for', 'with', 'this', 'from', 'are', 'have']
        english_count = sum(1 for word in english_indicators if f' {word} ' in f' {sample} ')
        if english_count > 5:
            return Language.ENGLISH
        return Language.UNKNOWN

    def extract_metadata_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        metadata = {}
        try:
            reader = PdfReader(pdf_path)
            if reader.metadata:
                metadata.update({
                    'title': reader.metadata.get('/Title', ''),
                    'author': reader.metadata.get('/Author', ''),
                    'creation_date': reader.metadata.get('/CreationDate', ''),
                })
            metadata['total_pages'] = len(reader.pages)
            metadata['file_size'] = Path(pdf_path).stat().st_size
        except Exception as e:
            logger.warning(f"Metadata error: {e}")
        return metadata

    def process_page(self, page_data: Tuple[int, Any], source_path: str) -> Optional[Document]:
        page_number, page = page_data
        try:
            text = page.extract_text() or page.extract_text(extraction_mode="layout")
            text = self.clean_text(text)
            if not text or len(text.strip()) < self.min_chunk_length: return None
            return Document(
                page_content=text,
                metadata={
                    "page_number": page_number + 1,
                    "language": self.detect_language(text).value,
                    "source_path": source_path
                }
            )
        except Exception as e:
            return None

    def extract_content_from_pdf(self, file_path: str) -> Tuple[List[Document], Dict[str, Any]]:
        documents = []
        try:
            reader = PdfReader(file_path)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.process_page, (i, page), file_path): i for i, page in
                           enumerate(reader.pages)}
                for future in as_completed(futures):
                    doc = future.result()
                    if doc: documents.append(doc)
            return documents, {"total_pages": len(reader.pages), "processed_pages": len(documents)}
        except Exception as e:
            raise e

    def get_separators_for_language(self, language: Language) -> List[str]:
        return self.separator_patterns.get(language, self.separator_patterns[Language.UNKNOWN])

    def intelligent_chunking(
            self,
            documents: List[Document],
            source_id: str,
            book_title: str,
            primary_language: Optional[Language] = None
    ) -> Tuple[List[Document], Dict[str, Any]]:
        all_docs = []
        stats = {"parents": 0, "children": 0}

        for i, doc in enumerate(documents):
            page_num = doc.metadata.get("page_number", i + 1)
            parent_id = f"{source_id}_p{page_num}"

            parent_doc = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "doc_type": "parent",
                    "doc_id": parent_id,
                    "source_id": source_id,
                    "book_title": book_title
                }
            )
            all_docs.append(parent_doc)
            stats["parents"] += 1

            doc_lang = Language(doc.metadata.get("language", "unknown"))
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.get_separators_for_language(doc_lang)
            )

            child_chunks = text_splitter.split_documents([parent_doc])
            for j, child in enumerate(child_chunks):
                child.metadata.update({
                    "doc_type": "child",
                    "parent_id": parent_id,
                    "chunk_id": f"{parent_id}_c{j}"
                })
                all_docs.append(child)
                stats["children"] += 1

        return all_docs, stats


def process_and_add_pdf(
        file_path: str,
        collection: Collection,
        source_id: str,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        detect_language: bool = True
) -> Tuple[List[str], Dict[str, Any]]:
    try:
        processor = PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        book_title = Path(file_path).stem

        pdf_metadata = processor.extract_metadata_from_pdf(file_path)
        raw_docs, _ = processor.extract_content_from_pdf(file_path)

        primary_lang = None
        if detect_language and pdf_metadata.get('detected_language'):
            try:
                primary_lang = Language(pdf_metadata['detected_language'])
            except:
                primary_lang = None

        final_docs, h_stats = processor.intelligent_chunking(
            documents=raw_docs,
            source_id=source_id,
            book_title=book_title,
            primary_language=primary_lang
        )

        ids, docs, metas = [], [], []
        for d in final_docs:
            uid = d.metadata["chunk_id"] if d.metadata["doc_type"] == "child" else d.metadata["doc_id"]
            ids.append(uid)
            docs.append(d.page_content)
            m = d.metadata.copy()
            m.update({"source": source_id, **pdf_metadata})
            metas.append(m)

        batch_size = 100
        for i in range(0, len(ids), batch_size):
            collection.upsert(
                documents=docs[i:i + batch_size],
                metadatas=metas[i:i + batch_size],
                ids=ids[i:i + batch_size]
            )

        return ids, {"stats": h_stats, "book": book_title}
    except Exception as e:
        logger.error(f"Error: {e}")
        return [], {"error": str(e)}
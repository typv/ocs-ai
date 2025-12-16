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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported languages enumeration"""
    ENGLISH = "en"
    VIETNAMESE = "vi"
    UNKNOWN = "unknown"


class PDFProcessor:
    def __init__(
            self,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            max_workers: int = 4,
            min_chunk_length: int = 50
    ):
        """
        Initialize PDF Processor with customizable parameters

        Args:
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
            max_workers: Number of parallel workers
            min_chunk_length: Minimum chunk length
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_workers = max_workers
        self.min_chunk_length = min_chunk_length

        # Language-specific separators
        self.separator_patterns = {
            Language.ENGLISH: [
                "\n\n# ",  # Main headings
                "\n\n## ",  # Subheadings
                "\n\n### ",
                "\n\n#### ",
                "\n\nChapter ",  # English chapter patterns
                "\n\nCHAPTER ",
                "\n\nSection ",
                "\n\nSECTION ",
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                ": ",
                ", ",
                " ",
                ""
            ],
            Language.VIETNAMESE: [
                "\n\n# ",
                "\n\n## ",
                "\n\n### ",
                "\n\n#### ",
                "\n\nChương ",  # Vietnamese chapter patterns
                "\n\nCHƯƠNG ",
                "\n\nPhần ",  # Vietnamese part patterns
                "\n\nPHẦN ",
                "\n\nMục ",
                "\n\nMỤC ",
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                ": ",
                ", ",
                " ",
                ""
            ],
            Language.UNKNOWN: [
                "\n\n# ",
                "\n\n## ",
                "\n\n### ",
                "\n\n#### ",
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                ": ",
                ", ",
                " ",
                ""
            ]
        }

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Normalize Unicode characters
        text = unicodedata.normalize('NFKC', text)

        # Replace special characters
        text = re.sub(r'\xa0', ' ', text)  # Non-breaking space
        text = re.sub(r'\r\n', '\n', text)  # Standardize newlines
        text = re.sub(r'\r', '\n', text)

        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)

        # Remove multiple newlines but keep paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Normalize punctuation spacing
        text = re.sub(r'([.,!?])(\S)', r'\1 \2', text)

        # Remove excessive hyphens and dashes
        text = re.sub(r'-{2,}', '—', text)  # Convert multiple hyphens to em dash

        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

        return text.strip()

    def detect_language(self, text: str) -> Language:
        """
        Detect language of text with improved accuracy

        Args:
            text: Text to analyze

        Returns:
            Detected language
        """
        if not text:
            return Language.UNKNOWN

        # Convert to lowercase for analysis
        sample = text.lower()

        # Check for Vietnamese characters
        vietnamese_chars = set('ăâđêôơưàáạảãầấậẩẫằắặẳẵèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ')
        if any(char in vietnamese_chars for char in sample[:500]):  # Check first 500 chars
            return Language.VIETNAMESE

        # Common English words and patterns
        english_indicators = [
            'the', 'and', 'that', 'for', 'with', 'this', 'from',
            'are', 'have', 'was', 'were', 'but', 'not', 'what',
            'all', 'were', 'when', 'your', 'can', 'said', 'there',
            'use', 'each', 'which', 'she', 'how', 'their', 'will',
            'other', 'about', 'out', 'many', 'then', 'them', 'these',
            'some', 'her', 'would', 'make', 'like', 'him', 'into',
            'time', 'has', 'look', 'two', 'more', 'write', 'go',
            'see', 'number', 'no', 'way', 'could', 'people',
            'my', 'than', 'first', 'water', 'been', 'call',
            'who', 'oil', 'its', 'now', 'find', 'long', 'down',
            'day', 'did', 'get', 'come', 'made', 'may', 'part'
        ]

        # Count English indicators
        english_count = sum(1 for word in english_indicators if f' {word} ' in f' {sample} ')

        # Check for English-specific punctuation and capitalization patterns
        english_patterns = [
            r'\b[A-Z][a-z]+\b',  # Proper nouns
            r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.',  # Titles
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b',
            r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b'
        ]

        pattern_count = sum(1 for pattern in english_patterns if re.search(pattern, text[:500]))

        # If we find many English indicators, classify as English
        if english_count > 10 or pattern_count > 3:
            return Language.ENGLISH

        # Default fallback
        return Language.UNKNOWN

    def extract_metadata_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract metadata from PDF file

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary containing metadata
        """
        metadata = {}
        try:
            reader = PdfReader(pdf_path)

            # Extract metadata from PDF info
            if reader.metadata:
                metadata.update({
                    'title': reader.metadata.get('/Title', ''),
                    'author': reader.metadata.get('/Author', ''),
                    'subject': reader.metadata.get('/Subject', ''),
                    'keywords': reader.metadata.get('/Keywords', ''),
                    'creator': reader.metadata.get('/Creator', ''),
                    'producer': reader.metadata.get('/Producer', ''),
                    'creation_date': reader.metadata.get('/CreationDate', ''),
                    'modification_date': reader.metadata.get('/ModDate', ''),
                })

            # Basic statistics
            metadata['total_pages'] = len(reader.pages)
            metadata['file_size'] = Path(pdf_path).stat().st_size

            # Try to determine language from metadata
            if metadata.get('title') or metadata.get('subject'):
                combined_text = f"{metadata.get('title', '')} {metadata.get('subject', '')}"
                detected_lang = self.detect_language(combined_text)
                metadata['detected_language'] = detected_lang.value

        except Exception as e:
            logger.warning(f"Failed to extract metadata from {pdf_path}: {e}")

        return metadata

    def process_page(self, page_data: Tuple[int, Any], source_path: str) -> Optional[Document]:
        """
        Process a single PDF page

        Args:
            page_data: Tuple containing (page_number, page_object)
            source_path: Path to source PDF

        Returns:
            Document or None if no text
        """
        page_number, page = page_data

        try:
            # Try multiple extraction methods
            text = page.extract_text()

            if not text or len(text.strip()) < self.min_chunk_length:
                # Try alternative extraction method
                text = page.extract_text(extraction_mode="layout")

            text = self.clean_text(text)

            if not text or len(text.strip()) < self.min_chunk_length:
                return None

            # Detect language for this page
            language = self.detect_language(text)

            return Document(
                page_content=text,
                metadata={
                    "page_number": page_number + 1,
                    "language": language.value,
                    "source_path": source_path,
                    "char_count": len(text),
                    "word_count": len(text.split())
                }
            )

        except Exception as e:
            logger.warning(f"Error processing page {page_number}: {e}")
            return None

    def extract_content_from_pdf(self, file_path: str) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Extract content from PDF with parallel processing

        Args:
            file_path: Path to PDF file

        Returns:
            Tuple of (list of Documents, document statistics)
        """
        documents = []
        stats = {
            "total_pages": 0,
            "processed_pages": 0,
            "language_distribution": {},
            "avg_words_per_page": 0
        }

        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            stats["total_pages"] = total_pages

            logger.info(f"Processing {total_pages} pages from {file_path}")

            # Process pages in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Prepare tasks
                futures = {
                    executor.submit(self.process_page, (i, page), file_path): i
                    for i, page in enumerate(reader.pages)
                }

                # Collect results
                for future in as_completed(futures):
                    page_idx = futures[future]
                    try:
                        doc = future.result(timeout=30)
                        if doc:
                            documents.append(doc)
                            stats["processed_pages"] += 1

                            # Update language statistics
                            lang = doc.metadata.get("language", "unknown")
                            stats["language_distribution"][lang] = \
                                stats["language_distribution"].get(lang, 0) + 1

                            # Log progress
                            if len(documents) % 10 == 0:
                                logger.info(f"Processed {len(documents)}/{total_pages} pages")

                    except Exception as e:
                        logger.error(f"Error processing page {page_idx}: {e}")

            # Calculate statistics
            if documents:
                total_words = sum(doc.metadata.get("word_count", 0) for doc in documents)
                stats["avg_words_per_page"] = total_words / len(documents)

            logger.info(f"Completed processing {len(documents)} pages with content")

        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            raise

        return documents, stats

    def get_separators_for_language(self, language: Language) -> List[str]:
        """
        Get appropriate separators for the given language

        Args:
            language: Language to get separators for

        Returns:
            List of separators
        """
        return self.separator_patterns.get(language, self.separator_patterns[Language.UNKNOWN])

    def intelligent_chunking(
            self,
            documents: List[Document],
            source_id: str,
            book_title: str,
            primary_language: Optional[Language] = None
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Intelligent document chunking based on structure and language

        Args:
            documents: List of original Documents
            source_id: Source identifier
            book_title: Book title
            primary_language: Primary language of the book

        Returns:
            Tuple of (list of chunks, chunking statistics)
        """
        all_chunks = []
        stats = {
            "total_chunks": 0,
            "chunks_by_language": {},
            "avg_chunk_size": 0,
            "chunks_per_page": []
        }

        # Determine primary language if not provided
        if primary_language is None:
            # Find most common language in documents
            lang_counts = {}
            for doc in documents:
                lang = Language(doc.metadata.get("language", "unknown"))
                lang_counts[lang] = lang_counts.get(lang, 0) + 1

            if lang_counts:
                primary_language = max(lang_counts.items(), key=lambda x: x[1])[0]
            else:
                primary_language = Language.UNKNOWN

        logger.info(f"Using {primary_language.value} as primary language for chunking")

        # Process each document
        for i, doc in enumerate(documents):
            try:
                # Get language-specific separators
                doc_language = Language(doc.metadata.get("language", "unknown"))
                separators = self.get_separators_for_language(doc_language)

                # Create text splitter for this document's language
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    length_function=len,
                    separators=separators,
                    keep_separator=True,
                    is_separator_regex=False
                )

                # Enhance document with additional metadata
                enhanced_doc = Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "source_id": source_id,
                        "book_title": book_title,
                        "original_page": doc.metadata.get("page_number", i + 1),
                        "primary_language": primary_language.value,
                        "document_language": doc_language.value,
                    }
                )

                # Split document into chunks
                chunks = text_splitter.split_documents([enhanced_doc])

                # Process metadata for each chunk
                for chunk_idx, chunk in enumerate(chunks):
                    # Calculate chunk metrics
                    chunk_text = chunk.page_content
                    word_count = len(chunk_text.split())
                    char_count = len(chunk_text)

                    # Update chunk metadata
                    chunk.metadata.update({
                        "chunk_id": f"{source_id}_p{doc.metadata.get('page_number', i + 1)}_c{chunk_idx}",
                        "chunk_index": chunk_idx,
                        "total_chunks_in_page": len(chunks),
                        "word_count": word_count,
                        "char_count": char_count,
                        "avg_word_length": char_count / word_count if word_count > 0 else 0,
                        "is_english": doc_language == Language.ENGLISH,
                        "is_vietnamese": doc_language == Language.VIETNAMESE,
                    })

                    # Update statistics
                    stats["chunks_by_language"][doc_language.value] = \
                        stats["chunks_by_language"].get(doc_language.value, 0) + 1

                all_chunks.extend(chunks)
                stats["chunks_per_page"].append(len(chunks))

            except Exception as e:
                logger.error(f"Error processing page {i}: {e}")
                continue

        # Filter chunks that are too short
        filtered_chunks = [
            chunk for chunk in all_chunks
            if len(chunk.page_content.strip()) >= self.min_chunk_length
        ]

        # Update statistics
        stats["total_chunks"] = len(filtered_chunks)
        if filtered_chunks:
            total_chars = sum(chunk.metadata.get("char_count", 0) for chunk in filtered_chunks)
            stats["avg_chunk_size"] = total_chars / len(filtered_chunks)

        logger.info(f"Created {len(filtered_chunks)} chunks from {len(documents)} pages")
        logger.info(f"Language distribution: {stats['chunks_by_language']}")

        return filtered_chunks, stats

    def generate_chunk_id(self, source_id: str, chunk_index: int, content: str) -> str:
        """
        Generate unique ID for chunk

        Args:
            source_id: Source identifier
            chunk_index: Chunk index
            content: Chunk content

        Returns:
            Unique identifier
        """
        # Create hash from content to ensure uniqueness
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{source_id}_ch{chunk_index}_{content_hash}"


def process_and_add_pdf(
        file_path: str,
        collection: Collection,
        source_id: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        detect_language: bool = True
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Advanced PDF processing: Read PDF, extract text, intelligent chunking,
    and add to ChromaDB collection with support for multiple languages.

    Args:
        file_path: Path to PDF file
        collection: ChromaDB collection
        source_id: Unique source identifier
        chunk_size: Chunk size
        chunk_overlap: Chunk overlap
        detect_language: Whether to perform language detection

    Returns:
        Tuple of (list of chunk IDs, processing statistics)
    """
    try:
        # Initialize processor
        processor = PDFProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        book_title = Path(file_path).stem

        # Extract metadata
        logger.info(f"Extracting metadata from {file_path}")
        pdf_metadata = processor.extract_metadata_from_pdf(file_path)

        # Extract content
        logger.info(f"Extracting content from {file_path}")
        raw_documents, extraction_stats = processor.extract_content_from_pdf(file_path)

        if not raw_documents:
            logger.warning(f"PDF {file_path} contains no readable text")
            return [], {"error": "No readable text found"}

        # Determine primary language from metadata or detection
        primary_language = None
        if detect_language and pdf_metadata.get('detected_language'):
            try:
                primary_language = Language(pdf_metadata['detected_language'])
            except ValueError:
                primary_language = None

        # Intelligent chunking
        logger.info("Performing intelligent chunking...")
        final_chunks, chunking_stats = processor.intelligent_chunking(
            documents=raw_documents,
            source_id=source_id,
            book_title=book_title,
            primary_language=primary_language
        )

        # Filter very short chunks
        final_chunks = [
            chunk for chunk in final_chunks
            if len(chunk.page_content.strip()) >= processor.min_chunk_length
        ]

        logger.info(f"Chunking completed: {len(final_chunks)} chunks")

        # Prepare data for ChromaDB
        doc_ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(final_chunks):
            # Generate unique ID
            chunk_id = processor.generate_chunk_id(source_id, i, chunk.page_content)
            doc_ids.append(chunk_id)
            documents.append(chunk.page_content)

            # Prepare metadata
            chunk_metadata = chunk.metadata.copy()
            chunk_metadata.update({
                "source": source_id,
                "file_path": file_path,
                "book_title": book_title,
                "processing_timestamp": datetime.datetime.now().isoformat(),
                **pdf_metadata
            })
            metadatas.append(chunk_metadata)

        # Add to collection with batch processing
        batch_size = 100
        successful_inserts = 0

        for i in range(0, len(doc_ids), batch_size):
            batch_ids = doc_ids[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]

            try:
                collection.upsert(
                    documents=batch_docs,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                successful_inserts += len(batch_ids)
                logger.info(f"Added batch {i // batch_size + 1}: {len(batch_ids)} documents")

            except Exception as e:
                logger.error(f"Error adding batch {i // batch_size + 1}: {e}")
                # Try inserting individually
                for j in range(len(batch_ids)):
                    try:
                        collection.upsert(
                            documents=[batch_docs[j]],
                            metadatas=[batch_metadatas[j]],
                            ids=[batch_ids[j]]
                        )
                        successful_inserts += 1
                    except Exception as single_e:
                        logger.error(f"Error with document {batch_ids[j]}: {single_e}")

        # Compile final statistics
        final_stats = {
            "source_id": source_id,
            "book_title": book_title,
            "file_path": file_path,
            "total_pages": extraction_stats.get("total_pages", 0),
            "processed_pages": extraction_stats.get("processed_pages", 0),
            "total_chunks": chunking_stats.get("total_chunks", 0),
            "successful_inserts": successful_inserts,
            "language_distribution": extraction_stats.get("language_distribution", {}),
            "chunks_by_language": chunking_stats.get("chunks_by_language", {}),
            "avg_words_per_page": extraction_stats.get("avg_words_per_page", 0),
            "avg_chunk_size": chunking_stats.get("avg_chunk_size", 0),
            "pdf_metadata": pdf_metadata
        }

        logger.info(f"Completed: Added {successful_inserts} chunks from {source_id}")

        return doc_ids[:successful_inserts], final_stats

    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {e}", exc_info=True)
        return [], {"error": str(e)}
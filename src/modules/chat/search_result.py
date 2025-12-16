from datetime import datetime
from typing import Dict, Any

class SearchResult:
    """Container for search results with rich information"""

    def __init__(
            self,
            document: str,
            metadata: Dict[str, Any],
            score: float,
            source_id: str,
            doc_id: str,
            distance: float,
            search_strategy: str = "vector"
    ):
        self.document = document
        self.metadata = metadata
        self.score = score
        self.source_id = source_id
        self.doc_id = doc_id
        self.distance = distance
        self.search_strategy = search_strategy
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "document": self.document,
            "metadata": self.metadata,
            "score": self.score,
            "source_id": self.source_id,
            "doc_id": self.doc_id,
            "distance": self.distance,
            "search_strategy": self.search_strategy,
            "timestamp": self.timestamp
        }
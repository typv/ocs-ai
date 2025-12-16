import logging
import numpy as np

from typing import Optional, List, Dict, Any

from src.modules.chat.chat_enum import SearchStrategy
from src.modules.chat.search_result import SearchResult

logger = logging.getLogger(__name__)

class DocumentSearcher:
    """Advanced document searcher with multiple search strategies"""

    def __init__(
            self,
            collection,
            default_k: int = 20,
            default_strategy: SearchStrategy = SearchStrategy.FUSION,
            score_threshold: float = 0.3,
            enable_reranking: bool = True,
            rerank_top_n: int = 50
    ):
        """
        Initialize document searcher

        Args:
            collection: ChromaDB collection
            default_k: Default number of results to return
            default_strategy: Default search strategy
            score_threshold: Minimum similarity score threshold
            enable_reranking: Whether to enable reranking
            rerank_top_n: Number of documents to consider for reranking
        """
        self.collection = collection
        self.default_k = default_k
        self.default_strategy = default_strategy
        self.score_threshold = score_threshold
        self.enable_reranking = enable_reranking
        self.rerank_top_n = rerank_top_n
        self.search_history = []

    def distance_to_score(self, distance: float, method: str = "cosine") -> float:
        """
        Convert distance to similarity score

        Args:
            distance: Distance metric from vector search
            method: Distance method (cosine, euclidean, etc.)

        Returns:
            Similarity score between 0 and 1
        """
        if method == "cosine":
            # Cosine distance: 1 = similar, 0 = dissimilar
            return max(0.0, min(1.0, 1.0 - distance))
        elif method == "euclidean":
            # Euclidean distance: convert to similarity
            # Using exponential decay: score = exp(-distance)
            return min(1.0, np.exp(-distance / 10.0))
        else:
            # Default: assume distance is already similarity-like
            return max(0.0, min(1.0, 1.0 - distance))

    def language_aware_search(
            self,
            query: str,
            k: int = None,
            language_hint: Optional[str] = None,
            filter_by_source: Optional[List[str]] = None,
            filter_by_book: Optional[List[str]] = None,
            filter_by_language: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Perform language-aware vector search

        Args:
            query: Search query
            k: Number of results
            language_hint: Language hint for filtering
            filter_by_source: Filter by source IDs
            filter_by_book: Filter by book titles
            filter_by_language: Filter by languages

        Returns:
            List of search results
        """
        if k is None:
            k = self.default_k

        # Build where filter
        where_filter = {}

        if filter_by_source:
            where_filter["source"] = {"$in": filter_by_source}

        if filter_by_book:
            where_filter["book_title"] = {"$in": filter_by_book}

        if filter_by_language:
            where_filter["language"] = {"$in": filter_by_language}
        elif language_hint:
            # If language hint provided, prioritize documents in that language
            where_filter["language"] = {"$in": [language_hint, "unknown"]}

        # Prepare search parameters
        search_k = k * 3 if self.enable_reranking else k

        try:
            # Perform vector search
            query_results = self.collection.query(
                query_texts=[query],
                n_results=search_k,
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )

            results = []
            if query_results.get("ids") and query_results["ids"][0]:
                for i in range(len(query_results["ids"][0])):
                    doc_id = query_results["ids"][0][i]
                    document = query_results["documents"][0][i]
                    metadata = query_results["metadatas"][0][i]
                    distance = query_results["distances"][0][i]

                    # Convert distance to score
                    score = self.distance_to_score(distance)

                    # Apply score threshold
                    if score >= self.score_threshold:
                        result = SearchResult(
                            document=document,
                            metadata=metadata,
                            score=score,
                            source_id=metadata.get("source", "unknown"),
                            doc_id=doc_id,
                            distance=distance
                        )
                        results.append(result)

            # Sort by score
            results.sort(key=lambda x: x.score, reverse=True)

            # Apply reranking if enabled
            if self.enable_reranking and len(results) > 1:
                results = self.rerank_results(query, results[:self.rerank_top_n])

            return results[:k]

        except Exception as e:
            logger.error(f"Error in language-aware search: {e}")
            return []

    def semantic_reranking(
            self,
            query: str,
            documents: List[str],
            method: str = "bm25"
    ) -> List[float]:
        """
        Perform semantic reranking of documents

        Args:
            query: Search query
            documents: List of documents to rerank
            method: Reranking method (bm25, tfidf, etc.)

        Returns:
            List of reranking scores
        """
        # Simple implementation - can be enhanced with more sophisticated methods
        query_terms = query.lower().split()

        scores = []
        for doc in documents:
            doc_lower = doc.lower()
            score = 0

            # Simple term frequency-based scoring
            for term in query_terms:
                if len(term) > 2:  # Ignore very short terms
                    score += doc_lower.count(term) * len(term)

            # Normalize by document length
            if len(doc) > 0:
                score = score / (len(doc) ** 0.5)

            scores.append(score)

        # Normalize scores to 0-1 range
        if scores and max(scores) > 0:
            scores = [s / max(scores) for s in scores]

        return scores

    def rerank_results(
            self,
            query: str,
            results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Rerank search results using semantic scoring

        Args:
            query: Original query
            results: Initial search results

        Returns:
            Reranked results
        """
        if not results:
            return results

        # Extract documents for reranking
        documents = [r.document for r in results]

        # Get semantic reranking scores
        rerank_scores = self.semantic_reranking(query, documents)

        # Combine vector similarity and semantic scores
        for i, result in enumerate(results):
            if i < len(rerank_scores):
                # Weighted combination: 70% vector similarity, 30% semantic
                combined_score = (0.7 * result.score) + (0.3 * rerank_scores[i])
                result.score = combined_score

        # Resort by combined score
        results.sort(key=lambda x: x.score, reverse=True)

        return results

    def search_multiple_queries_fusion(
            self,
            queries: List[str],
            k: int = None,
            strategy: SearchStrategy = None,
            **kwargs
    ) -> List[SearchResult]:
        """
        Perform search with multiple queries and combine results (RAG-Fusion style)

        Args:
            queries: List of search queries (could be rewritten versions)
            k: Number of results
            strategy: Search strategy to use
            **kwargs: Additional search parameters

        Returns:
            Combined and deduplicated search results
        """
        if k is None:
            k = self.default_k

        if strategy is None:
            strategy = self.default_strategy

        if strategy == SearchStrategy.SIMPLE:
            # Simple search with first query only
            return self.language_aware_search(queries[0], k, **kwargs)

        # For fusion strategies, search with each query
        all_results = []

        for query in queries:
            # Get more results for each query to have enough for fusion
            query_results = self.language_aware_search(
                query,
                k=k * 2,  # Get more results per query
                **kwargs
            )
            all_results.extend(query_results)

        # Remove duplicates and keep best score per document
        unique_results = {}
        for result in all_results:
            if result.doc_id not in unique_results:
                unique_results[result.doc_id] = result
            elif result.score > unique_results[result.doc_id].score:
                unique_results[result.doc_id] = result

        # Sort by score and take top k
        sorted_results = sorted(
            unique_results.values(),
            key=lambda x: x.score,
            reverse=True
        )[:k]

        # Apply strategy-specific adjustments
        if strategy == SearchStrategy.RERANK:
            # Rerank the combined results
            if len(queries) > 0:
                combined_docs = [r.document for r in sorted_results]
                rerank_scores = self.semantic_reranking(queries[0], combined_docs)

                for i, result in enumerate(sorted_results):
                    if i < len(rerank_scores):
                        result.score = (0.6 * result.score) + (0.4 * rerank_scores[i])

                sorted_results.sort(key=lambda x: x.score, reverse=True)

        return sorted_results

    def search_with_filters(
            self,
            query: str,
            k: int = None,
            filters: Optional[Dict[str, Any]] = None,
            **kwargs
    ) -> List[SearchResult]:
        """
        Search with advanced filtering options

        Args:
            query: Search query
            k: Number of results
            filters: Dictionary of filters to apply
            **kwargs: Additional parameters

        Returns:
            Filtered search results
        """
        if k is None:
            k = self.default_k

        # Extract filters
        source_filter = None
        book_filter = None
        language_filter = None
        date_filter = None
        page_filter = None

        if filters:
            source_filter = filters.get("source_ids")
            book_filter = filters.get("book_titles")
            language_filter = filters.get("languages")
            date_filter = filters.get("date_range")
            page_filter = filters.get("page_range")

        # Build where clause for ChromaDB
        where_clause = {}

        if source_filter:
            where_clause["source"] = {"$in": source_filter}

        if book_filter:
            where_clause["book_title"] = {"$in": book_filter}

        if language_filter:
            where_clause["language"] = {"$in": language_filter}

        if date_filter:
            # Assuming metadata has processing_timestamp
            start_date, end_date = date_filter
            where_clause["processing_timestamp"] = {
                "$gte": start_date.isoformat(),
                "$lte": end_date.isoformat()
            }

        # Perform search
        search_k = k * 2  # Get more results for post-filtering

        try:
            query_results = self.collection.query(
                query_texts=[query],
                n_results=search_k,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )

            results = []
            if query_results.get("ids") and query_results["ids"][0]:
                for i in range(len(query_results["ids"][0])):
                    doc_id = query_results["ids"][0][i]
                    document = query_results["documents"][0][i]
                    metadata = query_results["metadatas"][0][i]
                    distance = query_results["distances"][0][i]

                    # Apply page filter if specified
                    if page_filter:
                        page_num = metadata.get("page_number", 0)
                        min_page, max_page = page_filter
                        if not (min_page <= page_num <= max_page):
                            continue

                    score = self.distance_to_score(distance)

                    if score >= self.score_threshold:
                        result = SearchResult(
                            document=document,
                            metadata=metadata,
                            score=score,
                            source_id=metadata.get("source", "unknown"),
                            doc_id=doc_id,
                            distance=distance
                        )
                        results.append(result)

            results.sort(key=lambda x: x.score, reverse=True)

            return results[:k]

        except Exception as e:
            logger.error(f"Error in filtered search: {e}")
            return []

    def get_search_stats(self) -> Dict[str, Any]:
        """Get search statistics"""
        if not self.search_history:
            return {"total_searches": 0}

        total_searches = len(self.search_history)
        avg_results = sum(log["result_count"] for log in self.search_history) / total_searches

        return {
            "total_searches": total_searches,
            "avg_results_per_search": avg_results,
            "recent_searches": self.search_history[-10:]  # Last 10 searches
        }
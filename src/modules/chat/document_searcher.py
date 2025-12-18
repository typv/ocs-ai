import logging
from typing import Optional, List, Dict, Any

from src.config.app_config import settings
from src.modules.chat.chat_enum import SearchStrategy
from src.modules.chat.search_result import SearchResult

logger = logging.getLogger(__name__)


class DocumentSearcher:
    def __init__(
            self,
            collection,
            default_k: int = settings.RAG.RETRIEVAL_K,
            default_strategy: SearchStrategy = SearchStrategy.FUSION,
            score_threshold: float = 0.3,
            enable_reranking: bool = True,
            rerank_top_n: int = 15
    ):
        self.collection = collection
        self.default_k = default_k
        self.default_strategy = default_strategy
        self.score_threshold = score_threshold
        self.enable_reranking = enable_reranking
        self.rerank_top_n = rerank_top_n

    def distance_to_score(self, distance: float, method: str = "cosine") -> float:
        if method == "cosine":
            return max(0.0, min(1.0, 1.0 - distance))
        return max(0.0, min(1.0, 1.0 - distance))

    def retrieve_parent_documents(self, child_results: List[SearchResult]) -> List[SearchResult]:
        if not child_results:
            return []

        parent_info = {}
        for res in child_results:
            p_id = res.metadata.get("parent_id")
            if p_id and p_id not in parent_info:
                parent_info[p_id] = {
                    "score": res.score,
                    "distance": res.distance
                }

        if not parent_info:
            return child_results

        try:
            parent_ids = list(parent_info.keys())
            parents = self.collection.get(
                ids=parent_ids,
                include=["documents", "metadatas"]
            )

            final_results = []
            for i in range(len(parents["ids"])):
                p_id = parents["ids"][i]
                final_results.append(SearchResult(
                    document=parents["documents"][i],
                    metadata=parents["metadatas"][i],
                    score=parent_info[p_id]["score"],
                    source_id=parents["metadatas"][i].get("source", "unknown"),
                    doc_id=p_id,
                    distance=parent_info[p_id]["distance"]
                ))

            final_results.sort(key=lambda x: x.score, reverse=True)
            return final_results
        except Exception as e:
            logger.error(f"Error retrieving parents: {e}")
            return child_results

    def search(
            self,
            queries: List[str],
            k: int = None,
            filters: Optional[Dict[str, Any]] = None,
            **kwargs
    ) -> List[SearchResult]:
        """
        Unified search function supporting Multi-query Fusion, Metadata Filtering,
        Score Boosting, and Hierarchical Parent Retrieval.
        """
        if k is None: k = self.default_k
        if isinstance(queries, str):
            queries = [queries]

        where_clause = {"doc_type": "child"}
        if filters:
            if filters.get("source_ids"): where_clause["source"] = {"$in": filters["source_ids"]}
            if filters.get("book_titles"): where_clause["book_title"] = {"$in": filters["book_titles"]}
            if filters.get("languages"): where_clause["language"] = {"$in": filters["languages"]}

        if kwargs.get("filter_by_source"):
            where_clause["source"] = {"$in": kwargs["filter_by_source"]}
        if kwargs.get("filter_by_book"):
            where_clause["book_title"] = {"$in": kwargs["filter_by_book"]}

        try:
            num_queries = len(queries)
            n_results_per_query = k * 5 if num_queries > 1 else k * 2

            query_results = self.collection.query(
                query_texts=queries,
                n_results=n_results_per_query,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )

            child_scores = {}
            child_data = {}

            for query_idx in range(len(query_results["ids"])):
                for i in range(len(query_results["ids"][query_idx])):
                    c_id = query_results["ids"][query_idx][i]
                    dist = query_results["distances"][query_idx][i]
                    score = self.distance_to_score(dist)

                    if score >= self.score_threshold:
                        if c_id not in child_scores:
                            child_scores[c_id] = score
                            child_data[c_id] = {
                                "doc": query_results["documents"][query_idx][i],
                                "meta": query_results["metadatas"][query_idx][i],
                                "dist": dist
                            }
                        else:
                            child_scores[c_id] += (score * 0.3)

            fused_children = [
                SearchResult(
                    document=child_data[c_id]["doc"],
                    metadata=child_data[c_id]["meta"],
                    score=final_score,
                    source_id=child_data[c_id]["meta"].get("source", "unknown"),
                    doc_id=c_id,
                    distance=child_data[c_id]["dist"]
                )
                for c_id, final_score in child_scores.items()
            ]

            parent_results = self.retrieve_parent_documents(fused_children)
            parent_results.sort(key=lambda x: x.score, reverse=True)

            return parent_results[:k]

        except Exception as e:
            logger.error(f"Unified Fusion Search Error: {e}")
            return []

    def get_search_stats(self) -> Dict[str, Any]:
        return {"status": "active", "mode": "hierarchical"}

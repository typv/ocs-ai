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
            default_k: int = 5,
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

    def language_aware_search(
            self,
            query: str,
            k: int = settings.RAG.RETRIEVAL_K,
            **kwargs
    ) -> List[SearchResult]:
        if k is None: k = self.default_k
        where_filter = {"doc_type": "child"}

        if kwargs.get("filter_by_source"):
            where_filter["source"] = {"$in": kwargs["filter_by_source"]}
        if kwargs.get("filter_by_book"):
            where_filter["book_title"] = {"$in": kwargs["filter_by_book"]}

        try:
            query_results = self.collection.query(
                query_texts=[query],
                n_results=k * 2,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )

            child_results = []
            if query_results.get("ids") and query_results["ids"][0]:
                for i in range(len(query_results["ids"][0])):
                    dist = query_results["distances"][0][i]
                    score = self.distance_to_score(dist)

                    if score >= self.score_threshold:
                        child_results.append(SearchResult(
                            document=query_results["documents"][0][i],
                            metadata=query_results["metadatas"][0][i],
                            score=score,
                            source_id=query_results["metadatas"][0][i].get("source", "unknown"),
                            doc_id=query_results["ids"][0][i],
                            distance=dist
                        ))

            return self.retrieve_parent_documents(child_results)[:k]
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def search_with_filters(
            self,
            query: str,
            k: int = settings.RAG.RETRIEVAL_K,
            filters: Optional[Dict[str, Any]] = None,
            **kwargs
    ) -> List[SearchResult]:
        if k is None: k = self.default_k
        where_clause = {"doc_type": "child"}

        if filters:
            if filters.get("source_ids"): where_clause["source"] = {"$in": filters["source_ids"]}
            if filters.get("book_titles"): where_clause["book_title"] = {"$in": filters["book_titles"]}
            if filters.get("languages"): where_clause["language"] = {"$in": filters["languages"]}

        try:
            query_results = self.collection.query(
                query_texts=[query],
                n_results=k * 2,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )

            child_results = []
            if query_results.get("ids") and query_results["ids"][0]:
                for i in range(len(query_results["ids"][0])):
                    dist = query_results["distances"][0][i]
                    score = self.distance_to_score(dist)
                    if score >= self.score_threshold:
                        child_results.append(SearchResult(
                            document=query_results["documents"][0][i],
                            metadata=query_results["metadatas"][0][i],
                            score=score,
                            source_id=query_results["metadatas"][0][i].get("source", "unknown"),
                            doc_id=query_results["ids"][0][i],
                            distance=dist
                        ))
            return self.retrieve_parent_documents(child_results)[:k]
        except Exception as e:
            logger.error(f"Error in filtered search: {e}")
            return []

    def search_multiple_queries_fusion(
            self,
            queries: List[str],
            k: int = settings.RAG.RETRIEVAL_K,
            **kwargs
    ) -> List[SearchResult]:
        if k is None: k = self.default_k
        all_results = []
        for q in queries:
            res = self.language_aware_search(q, k=k * 2, **kwargs)
            all_results.extend(res)

        unique_parents = {}
        for res in all_results:
            if res.doc_id not in unique_parents or res.score > unique_parents[res.doc_id].score:
                unique_parents[res.doc_id] = res

        sorted_results = sorted(unique_parents.values(), key=lambda x: x.score, reverse=True)
        return sorted_results[:k]

    def get_search_stats(self) -> Dict[str, Any]:
        return {"status": "active", "mode": "hierarchical"}
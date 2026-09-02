import re
import numpy as np
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document

class RAGMetricsEvaluator:
    """
    Evaluates RAG performance across three key dimensions:
    1. Faithfulness / Groundedness: Are claims in the answer supported by retrieved context?
    2. Answer Relevance: Does the answer directly address the user query?
    3. Context Precision: How relevant and high-quality are the top retrieved context chunks?
    """

    @staticmethod
    def _extract_keywords(text: str) -> set:
        """Extract meaningful normalized words/tokens."""
        words = re.findall(r'\b[a-zA-Z0-9_]{3,}\b', text.lower())
        # Filter out trivial stop words
        stopwords = {
            "the", "and", "is", "in", "it", "of", "to", "for", "with", "on", "that", "this",
            "are", "was", "were", "be", "has", "have", "had", "been", "which", "or", "an",
            "by", "as", "at", "from", "into", "about", "could", "would", "should", "what",
            "where", "when", "who", "how", "why", "can", "may", "answer", "question", "document"
        }
        return {w for w in words if w not in stopwords}

    def compute_faithfulness(self, answer: str, context_chunks: List[Document]) -> float:
        """
        Calculates Faithfulness / Groundedness score (0.0 to 1.0).
        Checks the fraction of key terms and claim tokens in answer present in retrieved context.
        """
        if not answer or not context_chunks:
            return 0.0

        if "could not find sufficient information" in answer.lower():
            # If the system correctly identified missing context, faithfulness to strict doc rule is 1.0
            return 1.0

        answer_keywords = self._extract_keywords(answer)
        if not answer_keywords:
            return 1.0

        context_text = " ".join([c.page_content for c in context_chunks]).lower()
        context_keywords = self._extract_keywords(context_text)

        # Count answer tokens found in context
        grounded_count = sum(1 for kw in answer_keywords if kw in context_keywords)
        faithfulness_score = grounded_count / len(answer_keywords)
        
        return round(float(np.clip(faithfulness_score * 1.1, 0.0, 1.0)), 2)

    def compute_answer_relevance(self, query: str, answer: str) -> float:
        """
        Calculates Answer Relevance score (0.0 to 1.0).
        Measures overlap between user query intent and answer body.
        """
        if not query or not answer:
            return 0.0

        query_keywords = self._extract_keywords(query)
        if not query_keywords:
            return 0.8  # Default baseline for short queries

        answer_keywords = self._extract_keywords(answer)
        
        # Check keyword coverage
        matched_q = sum(1 for kw in query_keywords if kw in answer_keywords)
        coverage = matched_q / len(query_keywords)
        
        # Length sanity bonus (penalize overly brief 1-word answers or blank responses)
        length_penalty = 1.0 if len(answer.split()) >= 5 else 0.5

        score = coverage * length_penalty
        return round(float(np.clip(score * 1.25, 0.1, 1.0)), 2)

    def compute_context_precision(self, retrieved_chunks: List[Tuple[Document, float]]) -> float:
        """
        Calculates Context Precision score (0.0 to 1.0) based on top-K chunk similarity scores.
        """
        if not retrieved_chunks:
            return 0.0

        scores = [score for _, score in retrieved_chunks]
        avg_precision = sum(scores) / len(scores)
        return round(float(np.clip(avg_precision, 0.0, 1.0)), 2)

    def evaluate_turn(
        self,
        query: str,
        answer: str,
        retrieved_chunks_with_scores: List[Tuple[Document, float]]
    ) -> Dict[str, Any]:
        """Runs full evaluation pipeline for a single RAG conversation turn."""
        chunks_only = [doc for doc, _ in retrieved_chunks_with_scores]
        
        faithfulness = self.compute_faithfulness(answer, chunks_only)
        relevance = self.compute_answer_relevance(query, answer)
        precision = self.compute_context_precision(retrieved_chunks_with_scores)
        
        overall_score = round(float((faithfulness * 0.4) + (relevance * 0.3) + (precision * 0.3)), 2)

        return {
            "faithfulness": faithfulness,
            "answer_relevance": relevance,
            "context_precision": precision,
            "overall_score": overall_score,
            "status": "PASS" if overall_score >= 0.5 else "WARN"
        }

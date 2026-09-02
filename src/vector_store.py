import os
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_OPTIONS

from langchain_core.embeddings import Embeddings

class GeminiEmbeddingWrapper(Embeddings):
    """Custom wrapper for Google Gemini embedding model."""
    def __init__(self, api_key: str, model_name: str = "models/text-embedding-004"):
        import google.generativeai as genai
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=api_key)
        self.genai = genai

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            res = self.genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(res['embedding'])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        res = self.genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_query"
        )
        return res['embedding']

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)

class TFIDFEmbeddingWrapper(Embeddings):
    """Zero-dependency fallback embedding wrapper using Scikit-Learn TF-IDF."""
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
        self.is_fitted = False

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        matrix = self.vectorizer.fit_transform(texts).toarray()
        self.is_fitted = True
        # Normalize vectors
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = (matrix / norms).tolist()
        return normalized

    def embed_query(self, text: str) -> List[float]:
        if not self.is_fitted:
            return [0.0] * 384
        vec = self.vectorizer.transform([text]).toarray()[0]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)

class VectorStoreManager:
    """Manages vector embeddings creation, indexing (FAISS/Chroma), and similarity retrieval."""
    
    def __init__(
        self,
        embedding_choice: str = "Local: sentence-transformers/all-MiniLM-L6-v2",
        vector_db_choice: str = "FAISS",
        api_key: str = ""
    ):
        self.embedding_choice = embedding_choice
        self.vector_db_choice = vector_db_choice
        self.api_key = api_key
        self.embeddings = self._init_embeddings()
        self.vector_store = None
        self.all_documents: List[Document] = []

    def _init_embeddings(self):
        """Initialize chosen embedding model with graceful fallbacks."""
        config = EMBEDDING_OPTIONS.get(self.embedding_choice, EMBEDDING_OPTIONS["Local: sentence-transformers/all-MiniLM-L6-v2"])
        emb_type = config["type"]
        model_name = config["model_name"]

        if emb_type == "huggingface":
            try:
                return HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            except Exception as e:
                print(f"HuggingFace embedding load warning ({e}), falling back to TF-IDF vectorizer.")
                return TFIDFEmbeddingWrapper()
        elif emb_type == "gemini":
            if not self.api_key:
                raise ValueError("Google Gemini API Key is required for Gemini embeddings.")
            return GeminiEmbeddingWrapper(api_key=self.api_key, model_name=model_name)
        elif emb_type == "openai":
            if not self.api_key:
                raise ValueError("OpenAI API Key is required for OpenAI embeddings.")
            from langchain_community.embeddings import OpenAIEmbeddings
            return OpenAIEmbeddings(openai_api_key=self.api_key, model=model_name)
        elif emb_type == "tfidf":
            return TFIDFEmbeddingWrapper()
        else:
            return TFIDFEmbeddingWrapper()

    def create_vector_store(self, documents: List[Document]):
        """Build vector index from documents."""
        if not documents:
            return None

        self.all_documents = documents
        
        if self.vector_db_choice == "FAISS":
            from langchain_community.vectorstores import FAISS
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        elif self.vector_db_choice == "Chroma":
            from langchain_community.vectorstores import Chroma
            self.vector_store = Chroma.from_documents(documents, self.embeddings)
        else:
            from langchain_community.vectorstores import FAISS
            self.vector_store = FAISS.from_documents(documents, self.embeddings)

        return self.vector_store

    def search(
        self,
        query: str,
        k: int = 4,
        score_threshold: float = 0.0,
        search_type: str = "similarity"
    ) -> List[Tuple[Document, float]]:
        """
        Perform vector search.
        Returns list of (Document, similarity_score) tuples where similarity_score is normalized in [0, 1].
        """
        if not self.vector_store:
            return []

        if search_type == "mmr":
            # Maximal Marginal Relevance
            docs = self.vector_store.max_marginal_relevance_search(query, k=k, fetch_k=k*3)
            # Dummy high scores for MMR rank
            results = [(doc, 1.0 - (idx * 0.05)) for idx, doc in enumerate(docs)]
        else:
            # Similarity search with score
            if self.vector_db_choice == "FAISS":
                # FAISS returns L2 distance (smaller is better) or Inner Product depending on normalization
                docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)
                results = []
                for doc, score in docs_and_scores:
                    # Convert FAISS distance to normalized similarity score (0 to 1)
                    # For normalized vectors, L2 distance d in [0, 2], cosine similarity = 1 - d^2 / 2
                    sim_score = max(0.0, min(1.0, 1.0 - (score / 2.0)))
                    results.append((doc, float(sim_score)))
            else:
                # Chroma or generic
                docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)
                results = []
                for doc, score in docs_and_scores:
                    # Normalize score if distance
                    sim_score = 1.0 / (1.0 + float(score)) if score > 1.0 else (1.0 - float(score))
                    sim_score = max(0.0, min(1.0, sim_score))
                    results.append((doc, sim_score))

        # Filter by threshold
        if score_threshold > 0:
            results = [item for item in results if item[1] >= score_threshold]

        return results

    def get_all_chunks(self) -> List[Document]:
        """Return all indexed document chunks."""
        return self.all_documents

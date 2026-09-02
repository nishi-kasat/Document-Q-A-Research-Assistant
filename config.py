import os
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# API Keys
DEFAULT_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Embedding Models
EMBEDDING_OPTIONS = {
    "Local: sentence-transformers/all-MiniLM-L6-v2": {
        "type": "huggingface",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "requires_key": False
    },
    "Local: Scikit-Learn TF-IDF (Ultra Fast)": {
        "type": "tfidf",
        "model_name": "tfidf-vectorizer",
        "dimension": 384,
        "requires_key": False
    },
    "Local: BAAI/bge-small-en-v1.5": {
        "type": "huggingface",
        "model_name": "BAAI/bge-small-en-v1.5",
        "dimension": 384,
        "requires_key": False
    },
    "Google Gemini: text-embedding-004": {
        "type": "gemini",
        "model_name": "models/text-embedding-004",
        "dimension": 768,
        "requires_key": True
    },
    "OpenAI: text-embedding-3-small": {
        "type": "openai",
        "model_name": "text-embedding-3-small",
        "dimension": 1536,
        "requires_key": True
    }
}

# LLM Options
LLM_PROVIDERS = ["Google Gemini", "OpenAI", "Offline / Mock Engine"]

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

OPENAI_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-3.5-turbo"
]

VECTOR_STORE_PROVIDERS = ["FAISS", "Chroma"]

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4
DEFAULT_SCORE_THRESHOLD = 0.35  # Cosine similarity minimum threshold for relevance

SYSTEM_PROMPT_STRICT = """You are an expert Intelligent Document Assistant.
Answer the user's question using ONLY the provided context chunks extracted from uploaded documents.

CRITICAL INSTRUCTIONS:
1. Base your answer strictly on the context chunks provided below.
2. If the context does not contain enough information to answer the question, state clearly: "I could not find sufficient information in the uploaded document(s) to answer this question." Do NOT fabricate or assume details not present in the text.
3. Keep your response clear, well-structured, precise, and professional.
4. When relevant, reference key facts directly from the context.

Context Chunks:
{context}

Question:
{question}

Answer:"""

SYSTEM_PROMPT_HYBRID = """You are an expert Document Research Assistant.
Below are context chunks extracted from uploaded documents relevant to the user's question.

INSTRUCTIONS:
1. Prioritize answering based on the provided document context chunks.
2. If the document context provides partial or no information, you may supplement with general knowledge, but you MUST explicitly declare:
   "[Note: Part of this answer relies on general knowledge as it was not fully detailed in the provided documents.]"
3. Be clear, precise, and helpful.

Context Chunks:
{context}

Question:
{question}

Answer:"""

QUERY_CONDENSE_PROMPT = """Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that captures the full context required to search a vector database.

Conversation History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""

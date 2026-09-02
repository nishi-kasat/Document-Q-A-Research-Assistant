import time
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document
from config import SYSTEM_PROMPT_STRICT, SYSTEM_PROMPT_HYBRID, QUERY_CONDENSE_PROMPT
from src.vector_store import VectorStoreManager
from src.evaluation import RAGMetricsEvaluator

class RAGChainManager:
    """Coordinates Retrieval-Augmented Generation (RAG) execution across vector search and LLMs."""

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        provider: str = "Google Gemini",
        model_name: str = "gemini-2.5-flash",
        api_key: str = "",
        strict_mode: bool = True,
        score_threshold: float = 0.30,
        top_k: int = 4
    ):
        self.vector_store_manager = vector_store_manager
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.strict_mode = strict_mode
        self.score_threshold = score_threshold
        self.top_k = top_k
        self.evaluator = RAGMetricsEvaluator()

    def condense_question(self, chat_history: List[Dict[str, str]], latest_question: str) -> str:
        """
        Rephrases follow-up user query using conversation history.
        """
        if not chat_history:
            return latest_question

        formatted_history = ""
        for turn in chat_history[-3:]:  # Use last 3 turns
            formatted_history += f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}\n"

        prompt = QUERY_CONDENSE_PROMPT.format(chat_history=formatted_history, question=latest_question)

        # Call LLM or simple heuristic for query reformulation
        try:
            if self.provider == "Google Gemini" and self.api_key:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel(self.model_name)
                res = model.generate_content(prompt)
                condensed = res.text.strip()
                if condensed:
                    return condensed
            elif self.provider == "OpenAI" and self.api_key:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key)
                res = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                condensed = res.choices[0].message.content.strip()
                if condensed:
                    return condensed
        except Exception:
            pass

        return latest_question

    def _call_llm(self, prompt: str) -> str:
        """Call selected LLM provider with graceful fallback if API key is missing."""
        if self.provider == "Google Gemini":
            if not self.api_key:
                mock_ans = self._mock_generation(prompt)
                return f"[Note: No API key provided for Google Gemini. Synthesizing answer from retrieved document context using Offline Engine. Enter an API key in the sidebar to call live Gemini models.]\n\n{mock_ans}"
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            return response.text

        elif self.provider == "OpenAI":
            if not self.api_key:
                mock_ans = self._mock_generation(prompt)
                return f"[Note: No API key provided for OpenAI. Synthesizing answer from retrieved document context using Offline Engine. Enter an API key in the sidebar to call live OpenAI models.]\n\n{mock_ans}"
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return response.choices[0].message.content

        else:
            # Offline / Mock Fallback Generator
            return self._mock_generation(prompt)

    def _mock_generation(self, prompt: str) -> str:
        """Offline synthesizer for demo/testing without API keys."""
        if "Context Chunks:" in prompt:
            parts = prompt.split("Context Chunks:")
            context_block = parts[1].split("Question:")[0].strip()
            question_block = parts[1].split("Question:")[1].split("Answer:")[0].strip() if "Question:" in parts[1] else ""
        else:
            context_block = prompt
            question_block = ""

        if not context_block or "No context chunks retrieved" in context_block:
            return "I could not find sufficient information in the uploaded document(s) to answer this question."

        lines = [line.strip() for line in context_block.split("\n") if line.strip() and not line.startswith("---")]
        combined_text = " ".join(lines)
        
        # Split into clean sentences
        import re
        raw_sentences = [s.strip() for s in combined_text.replace("\n", " ").split(".") if len(s.strip()) > 10]
        
        # Extract query keywords
        q_words = set(re.findall(r'\b[a-zA-Z0-9_]{3,}\b', question_block.lower()))
        stopwords = {"what", "about", "how", "why", "when", "where", "which", "is", "are", "the", "and", "for", "with", "from"}
        q_keywords = q_words - stopwords

        if q_keywords and raw_sentences:
            scored = []
            for s in raw_sentences:
                s_words = set(re.findall(r'\b[a-zA-Z0-9_]{3,}\b', s.lower()))
                matches = len(q_keywords.intersection(s_words))
                scored.append((matches, s))
            scored.sort(key=lambda x: x[0], reverse=True)
            
            matching_sentences = [s for score, s in scored if score > 0]
            selected = matching_sentences[:3] if matching_sentences else raw_sentences[:3]
        else:
            selected = raw_sentences[:3] if raw_sentences else ["Relevant document context retrieved."]

        formatted_summary = "\n".join([f"- {s}." if not s.endswith(".") else f"- {s}" for s in selected])

        return f"Based on the retrieved document context:\n\n{formatted_summary}"

    def query(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        search_type: str = "similarity"
    ) -> Dict[str, Any]:
        """
        Executes full RAG flow:
        1. Query reformulation
        2. Vector search & similarity threshold check
        3. Prompt construction & LLM generation
        4. Citations preparation & Evaluation
        """
        start_time = time.time()
        chat_history = chat_history or []

        # 1. Condense follow-up question if history exists
        standalone_question = self.condense_question(chat_history, question)

        # 2. Retrieve vector store results
        retrieved_items = self.vector_store_manager.search(
            query=standalone_question,
            k=self.top_k,
            score_threshold=0.0,  # Get raw top-k first to check max score
            search_type=search_type
        )

        max_score = max([score for _, score in retrieved_items]) if retrieved_items else 0.0
        out_of_doc = max_score < self.score_threshold

        # Filter by threshold if strict mode is active
        if self.strict_mode and out_of_doc:
            filtered_items = []
        else:
            filtered_items = [item for item in retrieved_items if item[1] >= self.score_threshold]
            if not filtered_items and retrieved_items and not self.strict_mode:
                filtered_items = retrieved_items[:2]  # Keep best match in non-strict mode

        # 3. Format Context
        if filtered_items:
            context_blocks = []
            citations = []
            for idx, (doc, score) in enumerate(filtered_items):
                source = doc.metadata.get("source", "Document")
                page = doc.metadata.get("page", 1)
                chunk_id = doc.metadata.get("chunk_id", f"chunk_{idx+1}")
                
                context_blocks.append(
                    f"--- Chunk {idx+1} [Source: {source}, Page {page}, Score: {score:.2f}] ---\n{doc.page_content}"
                )
                citations.append({
                    "id": chunk_id,
                    "source": source,
                    "page": page,
                    "score": round(score, 3),
                    "snippet": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
                    "full_text": doc.page_content
                })
            context_str = "\n\n".join(context_blocks)
        else:
            context_str = "No context chunks retrieved matching the relevance threshold."
            citations = []

        # 4. Strict vs Hybrid Prompt Selection
        if self.strict_mode and out_of_doc:
            answer = "I could not find sufficient information in the uploaded document(s) to answer this question."
        else:
            system_template = SYSTEM_PROMPT_STRICT if self.strict_mode else SYSTEM_PROMPT_HYBRID
            prompt = system_template.format(context=context_str, question=standalone_question)
            try:
                answer = self._call_llm(prompt)
            except Exception as e:
                answer = f"LLM Generation Error: {str(e)}"

        latency = round(time.time() - start_time, 2)

        # 5. Evaluate RAG metrics
        eval_metrics = self.evaluator.evaluate_turn(
            query=question,
            answer=answer,
            retrieved_chunks_with_scores=filtered_items
        )

        return {
            "question": question,
            "standalone_question": standalone_question,
            "answer": answer,
            "citations": citations,
            "out_of_document": out_of_doc,
            "max_score": round(max_score, 3),
            "latency_seconds": latency,
            "evaluation": eval_metrics
        }

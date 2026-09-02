import os
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    DEFAULT_GEMINI_API_KEY,
    DEFAULT_OPENAI_API_KEY,
    EMBEDDING_OPTIONS,
    LLM_PROVIDERS,
    GEMINI_MODELS,
    OPENAI_MODELS,
    VECTOR_STORE_PROVIDERS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_SCORE_THRESHOLD
)
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.rag_chain import RAGChainManager

# Streamlit Page Configuration
st.set_page_config(
    page_title="Intelligent Document Q&A Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Theme-Adaptive Light & Dark Mode Support)
st.markdown("""
<style>
    /* Theme-Adaptive Base Typography */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sleek Title & Subtitle */
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-size: 0.98rem;
        opacity: 0.75;
        margin-bottom: 1.5rem;
    }
    
    /* Theme-Adaptive Metric Cards */
    .metric-card {
        background-color: var(--secondary-background-color, rgba(128, 128, 128, 0.08));
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 1.1rem;
        text-align: center;
    }
    .metric-val {
        font-size: 1.7rem;
        font-weight: 700;
        color: #4f46e5;
    }
    .metric-lbl {
        font-size: 0.78rem;
        opacity: 0.75;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* Theme-Adaptive Citation Box */
    .citation-card {
        background-color: var(--secondary-background-color, rgba(128, 128, 128, 0.08));
        border-left: 4px solid #6366f1;
        border-radius: 6px;
        padding: 0.9rem;
        margin-top: 0.6rem;
        margin-bottom: 0.6rem;
    }
    .citation-header {
        font-size: 0.85rem;
        font-weight: 600;
        color: #4f46e5;
    }
    .citation-score {
        float: right;
        background: rgba(99, 102, 241, 0.15);
        color: #4f46e5;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 10px;
    }
    
    /* Adaptive Badges */
    .out-bounds-badge {
        background: rgba(239, 68, 68, 0.12);
        color: #dc2626;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.6rem;
    }
    .in-bounds-badge {
        background: rgba(34, 197, 94, 0.12);
        color: #16a34a;
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "eval_history" not in st.session_state:
    st.session_state.eval_history = []
if "processed_chunks" not in st.session_state:
    st.session_state.processed_chunks = []
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = None
if "vector_manager" not in st.session_state:
    st.session_state.vector_manager = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Sidebar - Settings & Document Management
with st.sidebar:
    st.title("Control Panel")
    
    st.markdown("---")
    st.subheader("API Key & Provider")
    
    provider = st.selectbox("LLM Provider", LLM_PROVIDERS, index=0)
    
    if provider == "Google Gemini":
        model_name = st.selectbox("Model", GEMINI_MODELS, index=0)
        api_key = st.text_input("Gemini API Key", value=DEFAULT_GEMINI_API_KEY, type="password")
    elif provider == "OpenAI":
        model_name = st.selectbox("Model", OPENAI_MODELS, index=0)
        api_key = st.text_input("OpenAI API Key", value=DEFAULT_OPENAI_API_KEY, type="password")
    else:
        model_name = "Offline-Engine"
        api_key = ""
        st.info("Running in Offline Mode with simulated generation.")

    st.markdown("---")
    st.subheader("RAG Configurations")
    
    embedding_choice = st.selectbox(
        "Embedding Model",
        list(EMBEDDING_OPTIONS.keys()),
        index=0
    )
    vector_db_choice = st.selectbox("Vector Database", VECTOR_STORE_PROVIDERS, index=0)
    
    chunk_size = st.slider("Chunk Size (chars)", 200, 2000, DEFAULT_CHUNK_SIZE, 100)
    chunk_overlap = st.slider("Chunk Overlap (chars)", 0, 500, DEFAULT_CHUNK_OVERLAP, 50)
    top_k = st.slider("Top K Retrieved Chunks", 1, 10, DEFAULT_TOP_K)
    score_threshold = st.slider("Relevance Score Threshold", 0.0, 1.0, DEFAULT_SCORE_THRESHOLD, 0.05)
    
    strict_mode = st.toggle("Strict Document-Only Mode", value=True, help="Refuse questions that fall outside document context.")
    search_type = st.radio("Search Strategy", ["similarity", "mmr"], index=0, format_func=lambda x: "Cosine Similarity" if x == "similarity" else "Maximal Marginal Relevance (MMR)")

    st.markdown("---")
    if st.button("Clear Chat & Memory", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.eval_history = []
        st.rerun()

# Main Application Layout
st.markdown('<div class="main-title">Intelligent Document Q&A & Research Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multi-Document RAG | Vector Search | Citations | Conversational History | Quality Metrics Evaluation</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Research & Chat Workspace",
    "Vector Index & Chunk Inspector",
    "RAG Quality Evaluation",
    "Architecture & System Info"
])

# ==================== TAB 1: Chat Workspace ====================
with tab1:
    col_left, col_right = st.columns([1, 2], gap="medium")
    
    # Left Column: Document Ingestion Panel
    with col_left:
        st.subheader("Document Ingestion")
        uploaded_files = st.file_uploader(
            "Upload PDFs or Text Documents",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True
        )
        
        if st.button("Process & Index Documents", type="primary", use_container_width=True):
            if not uploaded_files:
                st.warning("Please upload at least one PDF or text document.")
            else:
                with st.spinner("Processing documents, generating embeddings & building vector index..."):
                    try:
                        processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                        all_raw_docs = []
                        all_chunks = []
                        
                        for uploaded_file in uploaded_files:
                            file_bytes = uploaded_file.read()
                            raw_docs, chunks = processor.process_file(file_bytes, uploaded_file.name)
                            all_raw_docs.extend(raw_docs)
                            all_chunks.extend(chunks)

                        if not all_chunks:
                            st.error("No readable text could be extracted from the uploaded files.")
                        else:
                            st.session_state.processed_chunks = all_chunks
                            st.session_state.doc_stats = processor.get_document_stats(all_chunks)

                            # Build Vector Store
                            vec_manager = VectorStoreManager(
                                embedding_choice=embedding_choice,
                                vector_db_choice=vector_db_choice,
                                api_key=api_key
                            )
                            vec_manager.create_vector_store(all_chunks)
                            st.session_state.vector_manager = vec_manager

                            # Init RAG Chain
                            st.session_state.rag_chain = RAGChainManager(
                                vector_store_manager=vec_manager,
                                provider=provider,
                                model_name=model_name,
                                api_key=api_key,
                                strict_mode=strict_mode,
                                score_threshold=score_threshold,
                                top_k=top_k
                            )

                            st.success(f"Successfully indexed {len(all_chunks)} chunks from {len(uploaded_files)} document(s).")
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")

        # Document Stats Summary
        if st.session_state.doc_stats:
            stats = st.session_state.doc_stats
            st.markdown("---")
            st.markdown("### Index Overview")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{stats["total_chunks"]}</div><div class="metric-lbl">Total Chunks</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{stats["total_words"]}</div><div class="metric-lbl">Total Words</div></div>', unsafe_allow_html=True)
            
            st.markdown(f"**Average Chunk Size:** `{stats['avg_chunk_length']}` characters")
            st.markdown("**Indexed Files:**")
            for src in stats["sources"]:
                st.markdown(f"- `{src}`")

    # Right Column: Chat & Citations Area
    with col_right:
        st.subheader("Assistant Workspace")
        
        if not st.session_state.rag_chain:
            st.info("Upload and index your documents on the left panel to begin asking questions.")
        else:
            # Display Chat History
            for turn in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.markdown(turn["user"])
                
                with st.chat_message("assistant"):
                    if turn.get("out_of_doc"):
                        st.markdown('<div class="out-bounds-badge">Question Outside Document Context</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="in-bounds-badge">Grounded in Document Context</div>', unsafe_allow_html=True)

                    st.markdown(turn["assistant"])
                    
                    # Display Citations
                    citations = turn.get("citations", [])
                    if citations:
                        with st.expander(f"View {len(citations)} Cited Document Snippets"):
                            for c in citations:
                                st.markdown(f"""
                                <div class="citation-card">
                                    <div class="citation-header">Source: {c['source']} (Page {c['page']})
                                        <span class="citation-score">Match: {c['score']*100:.1f}%</span>
                                    </div>
                                    <p style="font-size:0.88rem; margin-top:6px; opacity:0.9;">"{c['snippet']}"</p>
                                </div>
                                """, unsafe_allow_html=True)

            # Chat Input Box
            if user_query := st.chat_input("Ask a question about your uploaded documents..."):
                with st.chat_message("user"):
                    st.markdown(user_query)

                with st.chat_message("assistant"):
                    with st.spinner("Retrieving context & generating answer..."):
                        # Re-sync parameters if changed in sidebar
                        st.session_state.rag_chain.provider = provider
                        st.session_state.rag_chain.model_name = model_name
                        st.session_state.rag_chain.api_key = api_key
                        st.session_state.rag_chain.strict_mode = strict_mode
                        st.session_state.rag_chain.score_threshold = score_threshold
                        st.session_state.rag_chain.top_k = top_k

                        # Prepare Chat History format for condensation
                        history_for_chain = [
                            {"user": t["user"], "assistant": t["assistant"]}
                            for t in st.session_state.chat_history
                        ]

                        # Execute RAG Chain
                        res = st.session_state.rag_chain.query(
                            question=user_query,
                            chat_history=history_for_chain,
                            search_type=search_type
                        )

                        # Output status badge
                        if res["out_of_document"]:
                            st.markdown('<div class="out-bounds-badge">Question Outside Document Context</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="in-bounds-badge">Grounded in Document Context</div>', unsafe_allow_html=True)

                        st.markdown(res["answer"])

                        # Render Citations
                        if res["citations"]:
                            with st.expander(f"View {len(res['citations'])} Cited Document Snippets"):
                                for c in res["citations"]:
                                    st.markdown(f"""
                                    <div class="citation-card">
                                        <div class="citation-header">Source: {c['source']} (Page {c['page']})
                                            <span class="citation-score">Match: {c['score']*100:.1f}%</span>
                                        </div>
                                        <p style="font-size:0.88rem; margin-top:6px; opacity:0.9;">"{c['snippet']}"</p>
                                    </div>
                                    """, unsafe_allow_html=True)

                        # Record turn history
                        turn_data = {
                            "user": user_query,
                            "assistant": res["answer"],
                            "citations": res["citations"],
                            "out_of_doc": res["out_of_document"],
                            "evaluation": res["evaluation"],
                            "latency": res["latency_seconds"]
                        }
                        st.session_state.chat_history.append(turn_data)
                        st.session_state.eval_history.append({
                            "Query": user_query,
                            "Faithfulness": res["evaluation"]["faithfulness"],
                            "Answer Relevance": res["evaluation"]["answer_relevance"],
                            "Context Precision": res["evaluation"]["context_precision"],
                            "Overall Score": res["evaluation"]["overall_score"],
                            "Latency (s)": res["latency_seconds"],
                            "Out Of Doc": res["out_of_document"]
                        })


# ==================== TAB 2: Vector Index Inspector ====================
with tab2:
    st.subheader("Vector Store & Chunk Explorer")
    
    if not st.session_state.processed_chunks:
        st.info("No indexed chunks available. Upload documents in Tab 1 first.")
    else:
        st.markdown("### Chunk Statistics & Distribution")
        
        chunk_df = pd.DataFrame([
            {
                "Chunk ID": c.metadata.get("chunk_id", f"chunk_{i}"),
                "Source": c.metadata.get("source", "Unknown"),
                "Page": c.metadata.get("page", 1),
                "Char Count": c.metadata.get("char_count", len(c.page_content)),
                "Word Count": c.metadata.get("word_count", len(c.page_content.split()))
            }
            for i, c in enumerate(st.session_state.processed_chunks)
        ])

        col_a, col_b = st.columns(2)
        with col_a:
            fig_hist = px.histogram(
                chunk_df,
                x="Char Count",
                nbins=15,
                title="Chunk Character Length Distribution",
                color_discrete_sequence=["#6366f1"]
            )
            fig_hist.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_b:
            fig_pie = px.pie(
                chunk_df,
                names="Source",
                title="Chunks per Document Source",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_pie.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")
        st.markdown("### Direct Vector Search Test")
        test_query = st.text_input("Run dry-run similarity search on vector index:")
        if test_query and st.session_state.vector_manager:
            search_res = st.session_state.vector_manager.search(
                query=test_query,
                k=top_k,
                score_threshold=0.0,
                search_type=search_type
            )
            
            st.markdown(f"**Retrieved {len(search_res)} matching chunks:**")
            for doc, score in search_res:
                st.markdown(f"""
                - **Document**: `{doc.metadata.get('source')}` | **Page**: {doc.metadata.get('page')} | **Cosine Similarity**: `{score:.3f}`
                > {doc.page_content[:250]}...
                """)

        st.markdown("---")
        st.markdown("### Raw Indexed Chunks Table")
        st.dataframe(chunk_df, use_container_width=True)


# ==================== TAB 3: RAG Quality Evaluation ====================
with tab3:
    st.subheader("RAG Quality & Evaluation Dashboard")
    
    if not st.session_state.eval_history:
        st.info("No evaluation data recorded yet. Ask questions in Tab 1 to generate evaluation metrics.")
    else:
        eval_df = pd.DataFrame(st.session_state.eval_history)
        
        # Calculate summary averages
        avg_faith = eval_df["Faithfulness"].mean() * 100
        avg_rel = eval_df["Answer Relevance"].mean() * 100
        avg_prec = eval_df["Context Precision"].mean() * 100
        avg_overall = eval_df["Overall Score"].mean() * 100

        st.markdown("### Aggregate Quality Metrics")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#0284c7;">{avg_faith:.1f}%</div><div class="metric-lbl">Faithfulness / Groundedness</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#7c3aed;">{avg_rel:.1f}%</div><div class="metric-lbl">Answer Relevance</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#16a34a;">{avg_prec:.1f}%</div><div class="metric-lbl">Context Precision</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#e11d48;">{avg_overall:.1f}%</div><div class="metric-lbl">Overall Quality Score</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        
        # Metrics Bar Chart
        fig_bar = px.bar(
            eval_df,
            x=eval_df.index + 1,
            y=["Faithfulness", "Answer Relevance", "Context Precision"],
            barmode="group",
            title="Per-Turn Metric Breakdown",
            labels={"x": "Question Turn Index", "value": "Score (0-1.0)"},
            color_discrete_sequence=["#0284c7", "#7c3aed", "#16a34a"]
        )
        fig_bar.update_layout(yaxis_range=[0, 1.05])
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        st.markdown("### Turn Evaluation History")
        st.dataframe(eval_df, use_container_width=True)

        # Export CSV Button
        csv_data = eval_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Export Evaluation Report (CSV)",
            data=csv_data,
            file_name="rag_evaluation_report.csv",
            mime="text/csv"
        )


# ==================== TAB 4: Architecture & Info ====================
with tab4:
    st.subheader("System Architecture & Workflow")
    
    st.markdown("""
    ### Technical Architecture

    ```
    ┌─────────────────────────┐      ┌───────────────────────────┐      ┌─────────────────────────┐
    │  Document Ingestion     │ ───► │  Recursive Chunking       │ ───► │  Embeddings Generator   │
    │    (PDF, TXT, MD)       │      │    (Custom Size/Overlap)  │      │   (HF / Gemini / OpenAI)│
    └─────────────────────────┘      └───────────────────────────┘      └────────────┬────────────┘
                                                                                     │
                                                                                     ▼
    ┌─────────────────────────┐      ┌───────────────────────────┐      ┌─────────────────────────┐
    │  Answer & Citations     │ ◄─── │  RAG LLM Generator        │ ◄─── │  Vector Database        │
    │    & Quality Metrics    │      │ (Gemini / OpenAI / Offline)│      │     (FAISS / Chroma)    │
    └─────────────────────────┘      └───────────────────────────┘      └─────────────────────────┘
    ```

    #### Key System Features:
    1. **Document Ingestion & Chunking**: Uses `pypdf` to parse documents page-by-page, appending exact page numbers and document titles to metadata.
    2. **Multi-Model Embeddings**: Supports HuggingFace `sentence-transformers/all-MiniLM-L6-v2` locally out of the box, as well as Gemini `text-embedding-004` and OpenAI `text-embedding-3-small`.
    3. **Vector Database**: Fast vector similarity search with FAISS or ChromaDB, supporting similarity thresholding and MMR re-ranking.
    4. **Conversational Memory**: Standalone query condensation using past conversation history turns.
    5. **Out-of-Document Detection**: Automatically evaluates retrieved chunk distance to flag questions that fall outside document bounds.
    6. **Automated RAG Evaluation Suite**: Computes real-time Faithfulness, Answer Relevance, and Context Precision scores for every query.
    """)

import os

import streamlit as st

from answer_engine import generate_answer, handle_smalltalk, llm_configured
from rag_engine import RAGEngine

# 1. Custom CSS Styling for Premium Aesthetics
st.set_page_config(
    page_title="SRM Admissions AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Deep Navy and Ice Blue palette styling
custom_css = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
    }
    
    /* Main Layout */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Custom Title Banner */
    .banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #0d9488 100%);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    
    .banner-title {
        color: #ffffff;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .banner-subtitle {
        color: #e2e8f0;
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    /* Chat Bubbles Style */
    .chat-bubble {
        padding: 1rem 1.5rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        max-width: 80%;
        line-height: 1.6;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .user-bubble {
        background-color: #21262d;
        border: 1px solid #30363d;
        color: #f0f6fc;
        margin-left: auto;
        border-bottom-right-corner: 4px;
    }
    
    .assistant-bubble {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #3b82f6;
        color: #f0f6fc;
        margin-right: auto;
        border-bottom-left-corner: 4px;
    }
    
    /* Metadata block styles */
    .source-box {
        background-color: #161b22;
        border-left: 4px solid #3b82f6;
        padding: 0.8rem;
        margin-top: 0.5rem;
        border-radius: 4px;
        font-size: 0.9rem;
        color: #8b949e;
    }
    
    /* Clean inputs */
    .stTextInput input {
        background-color: #0d1117 !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important;
        border-radius: 8px !important;
    }
    
    /* Custom buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: transform 0.1s ease !important;
    }
    .stButton>button:active {
        transform: scale(0.98);
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 2. Main Title Layout
st.markdown(
    """
    <div class="banner">
        <div class="banner-title">🎓 SRM Admissions AI Hub</div>
        <div class="banner-subtitle">Answers questions from the SRM Admission Brochure 2026-27 &amp; Hostel Circulars — offline by default, AI-powered when you add an API key</div>
    </div>
    """,
    unsafe_allow_html=True
)

# 3. Initialize the retrieval engine (loads the precomputed index; no model downloads)
if "rag_engine" not in st.session_state:
    with st.spinner("Loading brochure index..."):
        st.session_state.rag_engine = RAGEngine()
        st.session_state.rag_engine.build_or_load_index()
    st.toast("Index loaded successfully!", icon="✅")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Confidence threshold: below this, we say the brochure doesn't cover the question
MIN_SCORE = 0.15

# 4. Sidebar Controls
with st.sidebar:
    st.markdown("### 📚 Knowledge Base")
    engine = st.session_state.rag_engine
    doc_counts = {}
    for c in engine.chunks:
        doc_counts[c["doc_id"]] = doc_counts.get(c["doc_id"], 0) + 1

    for doc in engine.documents:
        st.markdown(f"**{doc['title']}**")
        st.caption(f"{doc_counts.get(doc['id'], 0)} indexed sections")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    if st.button("🔄 Rebuild PDF Index"):
        with st.spinner("Rebuilding document index..."):
            st.session_state.rag_engine.build_or_load_index(force_rebuild=True)
        st.toast("Index rebuilt successfully!", icon="✅")
        st.rerun()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 🤖 AI Answer Engine")
    if "llm_api_key" not in st.session_state:
        st.session_state.llm_api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    if "llm_base_url" not in st.session_state:
        st.session_state.llm_base_url = os.environ.get("LLM_BASE_URL") or "https://api.openai.com/v1"
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = os.environ.get("LLM_MODEL") or "gpt-4o-mini"

    llm_key = st.text_input(
        "API Key (optional)",
        value=st.session_state.llm_api_key,
        type="password",
        placeholder="sk-...",
        help="Leave empty to use the offline answer engine. Works with any OpenAI-compatible API (OpenAI, Groq, OpenRouter, Ollama...).",
    )
    llm_base = st.text_input("API Base URL", value=st.session_state.llm_base_url)
    llm_model = st.text_input("Model", value=st.session_state.llm_model)
    st.session_state.llm_api_key = llm_key.strip()
    st.session_state.llm_base_url = llm_base.strip()
    st.session_state.llm_model = llm_model.strip()

    if llm_configured({"api_key": st.session_state.llm_api_key}):
        st.success("✨ **LLM mode active** — answers are written by the AI model.")
    else:
        st.info("⚡ **Offline mode** — answers are composed locally from the brochure text. "
                "Add an API key above to enable AI-written answers.")

    st.markdown("---")
    st.info(
        "Answers are grounded in the **SRM Admission Brochure 2026-27** and "
        "**hostel circulars/fee structures**. The retrieval engine (TF-IDF + "
        "keyword search) finds the relevant sections, and the answer engine "
        "composes a direct reply from them."
    )

# 5. Chat History Display
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        # Display sources if present
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("🔍 View References"):
                for s_idx, src in enumerate(msg["sources"]):
                    chunk = src["chunk"]
                    score = src["score"]
                    st.markdown(
                        f"**Source {s_idx+1} ({chunk['doc_title']}, Page {chunk['page']})** | "
                        f"Similarity Score: `{score:.3f}`\n"
                        f"> {chunk['text']}\n"
                        f"---"
                    )

# 6. User Chat Input and Query Execution
if prompt := st.chat_input("Ask a doubt about the SRM brochure... (e.g. What are the B.Tech programs?)"):
    # Display user query
    with st.chat_message("user"):
        st.write(prompt)

    # Store query in session history
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching brochure and composing your answer..."):
            smalltalk = handle_smalltalk(prompt)
            engine = st.session_state.rag_engine
            if smalltalk:
                answer = smalltalk
                retrieved = []
            else:
                retrieved = engine.search(prompt, top_k=8)

            if not smalltalk and (not retrieved or retrieved[0]["score"] < MIN_SCORE):
                answer = (
                    "I couldn't find any relevant details in the brochure regarding your question. "
                    "Try asking about programs, entrance exams, scholarships, placements, or contact details."
                )
                retrieved = []
            elif not smalltalk:
                answer, _ = generate_answer(
                    prompt,
                    retrieved,
                    engine,
                    {
                        "api_key": st.session_state.llm_api_key,
                        "base_url": st.session_state.llm_base_url,
                        "model": st.session_state.llm_model,
                    },
                )

            st.write(answer)

            # Display references
            if retrieved:
                with st.expander("🔍 View References"):
                    for s_idx, src in enumerate(retrieved):
                        chunk = src["chunk"]
                        score = src["score"]
                        st.markdown(
                            f"**Source {s_idx+1} ({chunk['doc_title']}, Page {chunk['page']})** | "
                            f"Similarity Score: `{score:.3f}`\n"
                            f"> {chunk['text']}\n"
                            f"---"
                        )

            # Store assistant response and its source metadata in session history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": retrieved
            })

            # Re-run page to refresh layout nicely
            st.rerun()

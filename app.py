import os

import streamlit as st

from answer_engine import generate_answer, handle_smalltalk, llm_configured
from rag_engine import RAGEngine
from llm_local import is_running_on_streamlit_cloud

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
        <div class="banner-subtitle">Conversational AI Assistant for SRM Admission Brochure 2026-27 &amp; Hostel Circulars — powered by local Qwen offline LLM</div>
    </div>
    """,
    unsafe_allow_html=True
)

# 3. Initialize the retrieval engine (cached across reruns; loads precomputed index)
@st.cache_resource
def _load_rag_engine():
    engine = RAGEngine()
    engine.build_or_load_index()
    return engine

if "rag_engine" not in st.session_state:
    with st.spinner("Loading brochure index..."):
        st.session_state.rag_engine = _load_rag_engine()
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
    provider_choice = st.selectbox(
        "AI Provider",
        options=["Free Cloud Heuristic", "Google Gemini (Free API Key)", "Groq (Ultra-Fast Llama 3.3)", "OpenAI / Custom API", "Local Qwen (Offline)"],
        index=0 if is_running_on_streamlit_cloud() else 4
    )

    if provider_choice == "Google Gemini (Free API Key)":
        provider_name = "gemini"
        default_model = "gemini-2.0-flash"
    elif provider_choice == "Groq (Ultra-Fast Llama 3.3)":
        provider_name = "groq"
        default_model = "llama-3.3-70b-versatile"
    elif provider_choice == "OpenAI / Custom API":
        provider_name = "openai"
        default_model = "gpt-4o-mini"
    else:
        provider_name = "local"
        default_model = "Qwen2.5-0.5B"

    if provider_name != "local" and provider_choice != "Free Cloud Heuristic":
        api_key_input = st.text_input(
            f"{provider_choice} Key",
            value=os.environ.get(f"{provider_name.upper()}_API_KEY", ""),
            type="password",
            placeholder="Paste API Key here...",
        )
        st.session_state.llm_config = {
            "provider": provider_name,
            "api_key": api_key_input.strip(),
            "model": default_model
        }
    else:
        st.session_state.llm_config = None

    if is_running_on_streamlit_cloud() and not st.session_state.llm_config:
        st.info("☁️ **Cloud Mode Active** — Answers generated instantly by brochure parser. Add a free Gemini/Groq API key above to enable conversational AI.")
    elif st.session_state.llm_config and st.session_state.llm_config.get("api_key"):
        st.success(f"✨ **{provider_choice} Active** — powered by remote model.")
    else:
        st.info("⚡ **Offline Local LLM Active** — powered by local Qwen model.")

    st.markdown("---")
    st.info(
        "Grounded in **SRM Admission Brochure 2026-27** and **hostel circulars**."
    )

# 5. Main Multi-Tab Layout
tab_chat, tab_wizard, tab_compare = st.tabs([
    "💬 AI Admissions Assistant",
    "🎯 Program Eligibility Wizard",
    "⚖️ AI Course Comparer"
])

# ---------------------------------------------------------------------------
# TAB 1: AI Admissions Assistant Chat
# ---------------------------------------------------------------------------
with tab_chat:
    st.caption("Ask questions, explore course details, check fee structures, or click quick suggestions below.")

    # Quick Suggestion Pills
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    suggested_prompt = None
    if col_p1.button("🎓 B.Tech Majors 2026"):
        suggested_prompt = "What are the B.Tech programs available at SRM?"
    if col_p2.button("🏢 Boys Hostel Fees"):
        suggested_prompt = "First Year Boys Hostel Fees"
    if col_p3.button("📊 Placement Stats"):
        suggested_prompt = "What are the highest placement packages and total offers?"
    if col_p4.button("📞 Helpline Number"):
        suggested_prompt = "What is the admission helpline phone number?"

    # Audio input for voice queries
    audio_val = None
    if hasattr(st, "audio_input"):
        with st.expander("🎙️ Speak your Question (Voice Input)"):
            audio_val = st.audio_input("Record your question")

    # Render conversation history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
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

    # Determine input query (from chat_input or suggested button)
    prompt = st.chat_input("Ask a doubt about SRM admissions...") or suggested_prompt

    if prompt:
        with st.chat_message("user"):
            st.write(prompt)

        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Searching brochure and generating response..."):
                smalltalk = handle_smalltalk(prompt)
                engine = st.session_state.rag_engine
                if smalltalk:
                    answer = smalltalk
                    retrieved = []
                else:
                    is_broad = any(w in prompt.lower() for w in ["all", "list", "every", "complete", "what are the pg", "what are the ug"])
                    retrieved = engine.search(prompt, top_k=15 if is_broad else 8)

                if not smalltalk and (not retrieved or retrieved[0]["score"] < MIN_SCORE):
                    answer = (
                        "I couldn't find relevant details in the brochure regarding your question. "
                        "Try asking about programs, entrance exams, scholarships, placements, or contact details."
                    )
                    retrieved = []
                elif not smalltalk:
                    answer, _ = generate_answer(
                        prompt,
                        retrieved,
                        engine,
                        st.session_state.llm_config,
                        chat_history=st.session_state.chat_history
                    )

                st.write(answer)

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

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": retrieved
                })

    # Download Chat History as Markdown
    if st.session_state.chat_history:
        chat_text = "\n\n".join([f"**{m['role'].upper()}**: {m['content']}" for m in st.session_state.chat_history])
        st.download_button("📥 Export Conversation Transcript", data=chat_text, file_name="srm_admissions_chat.md", mime="text/markdown")


# ---------------------------------------------------------------------------
# TAB 2: AI Program Eligibility & Course Finder Wizard
# ---------------------------------------------------------------------------
with tab_wizard:
    st.subheader("🎯 Find Eligible SRMIST Programs")
    st.caption("Select your academic stream, score, and target level to view recommended programs, entrance exams, and campuses.")

    col1, col2, col3 = st.columns(3)
    with col1:
        user_stream = st.selectbox("Class 12 / UG Stream", ["PCM (Physics, Chem, Math)", "PCB (Physics, Chem, Bio)", "Commerce / Economics", "Arts / Humanities / Other"])
    with col2:
        user_score = st.slider("Aggregate Score (%)", min_value=40, max_value=100, value=75)
    with col3:
        user_degree = st.selectbox("Target Degree Level", ["UG (B.Tech, BBA, MBBS, B.Sc)", "PG (M.Tech, MBA, M.Sc, LL.M)"])

    if st.button("🔍 Check Eligible Programs", key="btn_eligibility"):
        engine = st.session_state.rag_engine
        level_code = "UG" if "UG" in user_degree else "PG"
        report = find_eligible_programs(user_stream, user_score, level_code, engine)
        st.markdown(report)


# ---------------------------------------------------------------------------
# TAB 3: AI Course Comparer
# ---------------------------------------------------------------------------
with tab_compare:
    st.subheader("⚖️ Compare SRMIST Degree Programs Side-by-Side")
    st.caption("Enter or select any two degree programs to generate an instant side-by-side comparison matrix.")

    col_a, col_b = st.columns(2)
    with col_a:
        prog_a = st.text_input("Program A", value="B.Tech Computer Science and Engineering")
    with col_b:
        prog_b = st.text_input("Program B", value="B.Tech Electronics and Communication Engineering")

    if st.button("⚖️ Compare Programs", key="btn_compare"):
        engine = st.session_state.rag_engine
        with st.spinner(f"Comparing {prog_a} vs {prog_b}..."):
            comp_report = compare_programs(prog_a, prog_b, engine)
            st.markdown(comp_report)


            # Re-run page to refresh layout nicely
            st.rerun()

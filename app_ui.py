import os
import time
import streamlit as st
from dotenv import load_dotenv
from src import services

# Load environment configuration
load_dotenv(override=True)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Pinecone & Groq RAG Assistant",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Cyber & Modern Dark Blue Theme Custom CSS
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Force Unified Dark Background Regardless of OS/Browser Mode */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div {
        background-color: #0B0F19 !important;
        color: #F8FAFC !important;
    }

    /* Thinking Box Animation */
    @keyframes pulseGlow {
        0% { opacity: 0.6; transform: scale(0.99); }
        50% { opacity: 1; transform: scale(1.01); }
        100% { opacity: 0.6; transform: scale(0.99); }
    }
    .thinking-box {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: rgba(30, 58, 138, 0.45);
        border: 1px solid rgba(59, 130, 246, 0.4);
        padding: 0.6rem 1.1rem;
        border-radius: 12px;
        color: #93C5FD;
        font-size: 0.92rem;
        font-weight: 600;
        animation: pulseGlow 1.6s infinite ease-in-out;
        margin-bottom: 0.75rem;
    }

    /* Hero Banner (Deep Blue Gradient) */
    .hero-box {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #2563EB 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: #FFFFFF;
        margin-bottom: 2rem;
        box-shadow: 0 12px 30px -5px rgba(37, 99, 235, 0.4);
        border: 1px solid rgba(59, 130, 246, 0.35);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(30, 58, 138, 0.6);
        backdrop-filter: blur(8px);
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
        color: #93C5FD;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.2;
        letter-spacing: -0.02em;
        color: #FFFFFF;
    }
    .hero-desc {
        font-size: 1.05rem;
        margin-top: 0.5rem;
        color: #CBD5E1;
        max-width: 780px;
        line-height: 1.5;
    }

    /* Vibrant Blue Action Buttons */
    div.stButton > button[kind="primary"],
    div.stButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        border: 1px solid rgba(96, 165, 250, 0.4) !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.4rem !important;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.45) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
        box-shadow: 0 6px 22px rgba(37, 99, 235, 0.65) !important;
        transform: translateY(-2px) !important;
        color: #FFFFFF !important;
    }

    div.stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* Sidebar Dark Styling */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B !important;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #F8FAFC !important;
    }

    /* Status Card in Sidebar (Dark Slate) */
    .status-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.1rem;
        margin-top: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .status-card-header {
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.6rem;
        font-size: 0.95rem;
    }
    .status-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 0;
        font-size: 0.88rem;
        color: #CBD5E1;
        border-bottom: 1px solid #334155;
    }
    .status-item:last-child {
        border-bottom: none;
    }
    .status-badge {
        background: #172554;
        color: #60A5FA;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
        border: 1px solid #1E40AF;
    }

    /* Chat Messages Dark Styling */
    [data-testid="stChatMessage"] {
        padding: 1.2rem 1.4rem;
        border-radius: 14px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
    }
    [data-testid="stChatMessage"] * {
        color: #F1F5F9 !important;
    }

    /* Assistant Message Specific Styling */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: linear-gradient(180deg, #1E293B 0%, #151F30 100%) !important;
        border-left: 4px solid #3B82F6 !important;
    }

    /* User Message Specific Styling */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(180deg, #172554 0%, #1E3A8A 100%) !important;
        border-left: 4px solid #60A5FA !important;
        border-color: #1E40AF !important;
    }

    /* Chat Input Styling (Bulletproof for Light & Dark Mode) */
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInputContainer"] {
        border-radius: 14px !important;
        background-color: #1E293B !important;
        border-color: #334155 !important;
    }
    [data-testid="stChatInput"] {
        border: 1.5px solid #334155 !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #F8FAFC !important;
        -webkit-text-fill-color: #F8FAFC !important;
        caret-color: #60A5FA !important;
        background-color: transparent !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #94A3B8 !important;
        -webkit-text-fill-color: #94A3B8 !important;
    }
    [data-testid="stChatInput"] button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.35) !important;
    }

    /* File Uploader Dropzone Styling (Bulletproof for Light & Dark Mode) */
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploader"] section {
        background-color: #1E293B !important;
        border: 2px dashed #3B82F6 !important;
        border-radius: 12px !important;
        color: #E2E8F0 !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover,
    [data-testid="stFileUploader"] section:hover {
        background-color: #172554 !important;
        border-color: #60A5FA !important;
    }
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] div {
        color: #CBD5E1 !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    /* Inputs & Textareas Styling */
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border: 1.5px solid #334155 !important;
        border-radius: 8px !important;
    }
    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.35) !important;
    }

    /* Tabs Styling in Sidebar */
    button[data-baseweb="tab"] {
        color: #94A3B8 !important;
        font-weight: 600 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #60A5FA !important;
        border-bottom-color: #3B82F6 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Hero Banner
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-badge">🌲 Pinecone Serverless • Groq LPU • Advanced RAG</div>
        <div class="hero-title">AI Knowledge Assistant</div>
        <div class="hero-desc">
            Upload your documents, ask questions, and experience real-time conversational streaming powered by Hybrid Vector Search & Cross-Encoder Reranking.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar: Document Ingestion & RAG Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📂 Document Ingestion")
    st.caption("Upload files or paste text to chunk and index into Pinecone.")

    upload_tab, text_tab = st.tabs(["📁 File Upload", "✍️ Paste Text"])

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Upload PDF, TXT or MD",
            type=["pdf", "txt", "md"],
            help="Upload files up to 25MB. Text is parsed into semantic overlapping chunks.",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            if st.button("🚀 Process & Index File", use_container_width=True, type="primary"):
                with st.spinner(f"Extracting and indexing '{uploaded_file.name}' into Pinecone..."):
                    try:
                        file_bytes = uploaded_file.read()
                        chunks_indexed, char_count = services.upload_and_index_file(
                            filename=uploaded_file.name,
                            content_bytes=file_bytes,
                        )
                        st.success(
                            f"✅ Successfully indexed **{chunks_indexed} chunks** ({char_count:,} characters)!"
                        )
                    except Exception as e:
                        st.error(f"❌ Failed to process document: {str(e)}")

    with text_tab:
        doc_title = st.text_input("Document Title", placeholder="e.g. NovaTech Policy 2026")
        doc_content = st.text_area(
            "Document Content",
            placeholder="Paste raw text, policies, or knowledge base content here...",
            height=140,
        )

        if st.button("🚀 Index Pasted Text", use_container_width=True, type="primary"):
            if not doc_title.strip():
                st.warning("⚠️ Please provide a document title.")
            elif not doc_content.strip():
                st.warning("⚠️ Document content cannot be empty.")
            else:
                with st.spinner("Chunking and indexing text into Pinecone..."):
                    try:
                        chunks_count = services.upload_and_index_document(
                            title=doc_title.strip(),
                            content=doc_content.strip(),
                        )
                        st.success(f"✅ Successfully indexed **{chunks_count} chunks**!")
                    except Exception as e:
                        st.error(f"❌ Failed to index document: {str(e)}")

    st.markdown("---")

    st.markdown("### ⚙️ Search Configuration")
    use_hybrid = st.toggle(
        "Hybrid Search & Reranking",
        value=True,
        help="Combines Dense Vector Search + BM25 Keyword Search + bge-reranker-v2-m3 for maximum precision.",
    )

    # Status Dashboard Card
    index_name = os.environ.get("PINECONE_INDEX_NAME", "rag-learn")
    active_model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
    redis_status = "Connected ⚡" if services.rag_cache.is_connected() else "Offline"

    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-card-header">
                ⚡ System Diagnostics
            </div>
            <div class="status-item">
                <span>Vector Index</span>
                <span class="status-badge">{index_name}</span>
            </div>
            <div class="status-item">
                <span>LLM Engine</span>
                <span class="status-badge">{active_model}</span>
            </div>
            <div class="status-item">
                <span>Search Mode</span>
                <span class="status-badge">{'Hybrid + Rerank' if use_hybrid else 'Dense Vector'}</span>
            </div>
            <div class="status-item">
                <span>Redis Cache</span>
                <span class="status-badge">{redis_status}</span>
            </div>
            <div class="status-item">
                <span>Streaming</span>
                <span class="status-badge">Enabled ⚡</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🗑️ Clear Conversation History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# -----------------------------------------------------------------------------
# Main Chat Interface (Session State & Streaming)
# -----------------------------------------------------------------------------
# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Welcome!** I am your AI Knowledge Assistant.\n\n"
                "• Upload a document (PDF, TXT) from the left sidebar.\n"
                "• Ask any question or follow-up inquiry to explore your data in real time!"
            ),
        }
    ]

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # 1. Display user prompt
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Append user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. Stream assistant response with visual thinking indicator and typewriter pacing
    with st.chat_message("assistant"):
        # Display thinking & searching indicator while Pinecone retrieves and reranks
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown(
            """
            <div class="thinking-box">
                <span>⚡</span> <span>Thinking and searching knowledge base...</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        try:
            prior_history = st.session_state.messages[:-1]

            # Generate real-time streaming tokens from the RAG service
            raw_generator = services.stream_conversational_rag(
                query=prompt,
                history=prior_history,
                use_hybrid=use_hybrid,
            )

            # Generator with typewriter pacing (18ms per token) so streaming is visibly smooth and natural
            def paced_stream_generator(gen):
                first_token = True
                for chunk in gen:
                    if first_token:
                        # Clear the thinking indicator as soon as the first token arrives
                        thinking_placeholder.empty()
                        first_token = False

                    # Split chunk into words for realistic typewriter streaming effect
                    words = chunk.split(" ")
                    for i, word in enumerate(words):
                        if i > 0:
                            yield " "
                        yield word
                        time.sleep(0.018)  # 18ms smooth delay

                # Clear thinking placeholder if it wasn't cleared
                thinking_placeholder.empty()

            # st.write_stream renders tokens live with the paced typing effect
            full_response = st.write_stream(paced_stream_generator(raw_generator))

            # 4. Save assistant response to session state
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            thinking_placeholder.empty()
            error_message = f"❌ An error occurred while generating the response: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

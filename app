"""
PDF Research Assistant — Streamlit Web App
===========================================
Author : Ramya Sree Nagarajan

Run:
  streamlit run app/app.py
"""

import streamlit as st
import sys, os, tempfile, time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.rag_engine import RAGPipeline, chunk_text, VectorStore, generate_answer_local

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Research Assistant",
    page_icon="📄",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
body, .main { background-color: #0D1117; color: #C9D1D9; }
.upload-box {
    border: 2px dashed #30363D;
    border-radius: 12px;
    padding: 30px;
    text-align: center;
    background: #161B22;
}
.answer-box {
    background: #161B22;
    border-left: 4px solid #70A5FD;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 15px;
    line-height: 1.7;
}
.chunk-card {
    background: #0D1117;
    border: 1px solid #21262D;
    border-radius: 8px;
    padding: 12px;
    margin: 6px 0;
    font-size: 13px;
    color: #8B949E;
}
.score-badge {
    background: #1F2937;
    color: #38BDAE;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: bold;
}
.metric-pill {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 10px 16px;
    text-align: center;
}
.stButton button {
    background: linear-gradient(135deg, #70A5FD, #BF91F3);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: bold;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "store"        not in st.session_state: st.session_state.store        = None
if "doc_name"     not in st.session_state: st.session_state.doc_name     = None
if "chunk_count"  not in st.session_state: st.session_state.chunk_count  = 0
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 📄 PDF Research Assistant")
st.markdown("**RAG-powered Q&A** — Upload any PDF and ask questions about its content")
st.divider()

# ── Layout ────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2])

# ── LEFT: Upload + Stats ──────────────────────────────────────────────────────
with col_left:
    st.markdown("### 📂 Upload Document")

    uploaded = st.file_uploader(
        "Drop your PDF here",
        type=["pdf"],
        label_visibility="collapsed"
    )

    # Demo mode button
    if st.button("🎬 Try Demo (no PDF needed)"):
        demo_text = """
        Artificial Intelligence in Healthcare.
        Machine learning algorithms achieve over 95 percent accuracy in detecting
        diabetic retinopathy. Deep learning models identify pneumonia and COVID-19
        from chest X-rays with over 90 percent sensitivity.
        NLP reduces hospital documentation time by 40 percent.
        Federated learning enables privacy-preserving collaborative training.
        The AI healthcare market will reach 187 billion dollars by 2030 at 37 percent CAGR.
        Reinforcement learning optimises chemotherapy dosing for cancer patients.
        """
        chunks = chunk_text(demo_text, chunk_size=60, overlap=10)
        store  = VectorStore()
        meta   = [{"chunk_id": i, "approx_page": 1, "source": "demo_healthcare.pdf"}
                   for i in range(len(chunks))]
        store.build(chunks, meta)
        st.session_state.store       = store
        st.session_state.doc_name    = "demo_healthcare.pdf"
        st.session_state.chunk_count = len(chunks)
        st.session_state.chat_history = []
        st.success("Demo document loaded!")

    if uploaded is not None and uploaded.name != st.session_state.doc_name:
        with st.spinner("📖 Reading and indexing PDF..."):
            try:
                import fitz
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                doc  = fitz.open(tmp_path)
                text = "\n".join(page.get_text() for page in doc)
                doc.close()
                os.unlink(tmp_path)
                chunks = chunk_text(text, chunk_size=400, overlap=80)
                store  = VectorStore()
                meta   = [{"chunk_id": i,
                            "approx_page": (i * 400) // 300 + 1,
                            "source": uploaded.name}
                           for i in range(len(chunks))]
                store.build(chunks, meta)
                st.session_state.store       = store
                st.session_state.doc_name    = uploaded.name
                st.session_state.chunk_count = len(chunks)
                st.session_state.chat_history = []
                st.success(f"✅ Indexed {len(chunks)} chunks!")
            except ImportError:
                st.error("PyMuPDF not installed. Run: pip install pymupdf")
            except Exception as e:
                st.error(f"Error reading PDF: {e}")

    # Document stats
    if st.session_state.store:
        st.divider()
        st.markdown("### 📊 Document Stats")
        st.markdown(f"""
        <div class="metric-pill">📄 <b>{st.session_state.doc_name}</b></div>
        <br/>
        <div class="metric-pill">🧩 <b>{st.session_state.chunk_count}</b> chunks indexed</div>
        <br/>
        <div class="metric-pill">💬 <b>{len(st.session_state.chat_history)}</b> questions asked</div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("### 💡 Sample Questions")
        sample_qs = [
            "What is the main topic?",
            "Summarise the key findings",
            "What methodology was used?",
            "What are the conclusions?",
        ]
        for q in sample_qs:
            if st.button(q, key=f"sq_{q}"):
                st.session_state._autofill = q

    # Sidebar info
    with st.sidebar:
        st.markdown("### ℹ️ How it works")
        st.markdown("""
        1. **Upload** any PDF document
        2. The app **chunks** text into overlapping segments
        3. Chunks are **embedded** using TF-IDF vectors
        4. Your question retrieves the **most relevant chunks**
        5. An answer is generated **from the document only**

        ---
        **Stack**
        - Python · Scikit-learn
        - TF-IDF + Cosine Similarity
        - PyMuPDF · Streamlit

        ---
        **Author**
        Ramya Sree Nagarajan
        MSc AI · IEEE Researcher
        """)

# ── RIGHT: Chat Interface ─────────────────────────────────────────────────────
with col_right:
    st.markdown("### 💬 Ask Questions")

    if st.session_state.store is None:
        st.info("👈 Upload a PDF or click **Try Demo** to get started")
    else:
        # Chat history
        for item in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(item["query"])
            with st.chat_message("assistant"):
                st.markdown(
                    f'<div class="answer-box">{item["answer"]}</div>',
                    unsafe_allow_html=True
                )
                with st.expander(f"📎 {len(item['chunks'])} source chunks retrieved"):
                    for i, (chunk, score) in enumerate(
                            zip(item["chunks"], item["scores"]), 1):
                        st.markdown(
                            f'<div class="chunk-card">'
                            f'<span class="score-badge">score {score:.3f}</span> '
                            f'<b>Chunk {i}</b><br/>{chunk[:300]}{"..." if len(chunk)>300 else ""}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

        # Auto-fill from sample question click
        default_q = getattr(st.session_state, "_autofill", "")
        if default_q:
            del st.session_state._autofill

        query = st.chat_input("Ask anything about your document...")

        if query:
            results = st.session_state.store.retrieve(query, top_k=5)
            chunks  = [r[0] for r in results]
            scores  = [r[1] for r in results]
            answer  = generate_answer_local(query, chunks)

            st.session_state.chat_history.append({
                "query":  query,
                "answer": answer,
                "chunks": chunks,
                "scores": scores,
            })
            st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#8B949E;font-size:13px;'>"
    "Built by <b>Ramya Sree Nagarajan</b> · MSc AI · Royal Holloway · "
    "IEEE Published Researcher · "
    "<a href='https://github.com/ramyasreenagarajan' style='color:#70A5FD;'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True
)

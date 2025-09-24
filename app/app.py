import os
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

# LOCAL embeddings (no API)
from langchain_community.embeddings import HuggingFaceEmbeddings

# LOCAL chat via Ollama (no API)
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA

# OPTIONAL: OpenAI chat if user brings a key
from langchain_openai import ChatOpenAI

# ---------------- Constants ----------------
DEFAULT_LOCAL_MODEL = "llama3.2:3b"   # also try "qwen2.5:3b"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# ---------------- UI Header ----------------
st.set_page_config(page_title="Political Narratives — Paper Q&A & Playground", layout="wide")
st.title("Political Narratives — Paper Q&A + Prompt Playground")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("Model settings")
    provider = st.selectbox(
        "Provider",
        ["Local (Ollama)", "OpenAI (bring your own key)"],
        index=0,
        key="provider_select",
    )
    temperature = st.slider(
        "Temperature", 0.0, 1.0, 0.2, 0.05, key="temperature_slider_sidebar"
    )

    local_model = DEFAULT_LOCAL_MODEL
    user_key = None

    if provider.startswith("Local"):
        st.markdown("### Local model (Ollama)")
        local_model = st.selectbox(
            "Ollama model",
            [DEFAULT_LOCAL_MODEL, "qwen2.5:3b", "llama3.2:3b"],
            index=0,
            key="ollama_model_select",
        )
        st.caption("Tip: keep it low for factual answers.")
    else:
        user_key = st.text_input(
            "OpenAI API key",
            type="password",
            help="Used only in your session. Leave empty to stay local.",
            key="openai_key_input",
        )

# ---------------- Helpers ----------------
def load_pdf(path: str):
    docs = []
    if not os.path.exists(path):
        return docs
    try:
        pdf = PdfReader(path)
        for i, p in enumerate(pdf.pages, start=1):
            text = p.extract_text() or ""
            docs.append(Document(page_content=text, metadata={"source": os.path.basename(path), "page": i}))
    except Exception as e:
        st.warning(f"Could not read PDF '{path}': {e}")
    return docs

def get_llm(provider: str, temperature: float, user_key: str | None, local_model: str):
    if provider.startswith("OpenAI") and user_key:
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature, api_key=user_key)
    return ChatOllama(model=local_model, temperature=temperature)

@st.cache_resource(show_spinner=True)
def build_vectorstore():
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_docs = []

    paper_path = "data/paper.pdf"  # <- rename your file to paper.pdf
    if os.path.exists(paper_path):
        for d in load_pdf(paper_path):
            for chunk in splitter.split_text(d.page_content or ""):
                all_docs.append(Document(page_content=chunk, metadata=d.metadata))
    else:
        st.warning("Paper not found at data/paper.pdf. Please add your PDF there.")

    if not all_docs:
        return None

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(all_docs, embeddings)

vs = build_vectorstore()
st.sidebar.write("Docs in index:", vs.index.ntotal if vs else 0)

# ---------------- UI Tabs ----------------
tab1, tab2 = st.tabs(["Ask about the paper (Local or OpenAI)", "Prompt playground"])

# ---------------- TAB 1: RAG Q&A ----------------
with tab1:
    if vs is None:
        st.info("No documents indexed. Add your paper at data/paper.pdf and rerun.")
    else:
        llm = get_llm(provider, temperature, user_key, local_model)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vs.as_retriever(search_kwargs={"k": 4}),
            chain_type="stuff",
            return_source_documents=False,  # don't return sources
        )
        st.caption("Ask questions about the paper. Local by default; switches to OpenAI if you paste a key.")
        q = st.text_input("Your question", key="qa_question_input")
        if q:
            with st.spinner("Thinking..."):
                out = qa({"query": q})
            st.write(out["result"])

# ---------------- TAB 2: Prompt playground ----------------
with tab2:
    st.caption("Try prompts locally with Ollama, or switch to OpenAI via the sidebar.")
    presets = {
        "Drama Triangle annotator (concise JSON)": (
            "You are an annotator using Drama Triangle roles: Victim, Rescuer, Persecutor. "
            "Be concise. First give a one-sentence explanation; then output a JSON object with this schema only: "
            "{ 'spans': [ { 'text': <substring>, 'role': 'victim|rescuer|persecutor', 'start': int, 'end': int } ] }."
        ),
        "Explain the framework (teacher style)": (
            "Explain the Drama Triangle framework clearly for a beginner in 5 bullet points, each <= 15 words."
        ),
        "Critique & improvements": (
            "Given the text, critique the analysis using Drama Triangle concepts; suggest 3 concrete improvements."
        ),
    }

    preset_name = st.selectbox("Preset", list(presets.keys()), key="preset_select")
    system_prompt = st.text_area("System / Instruction", presets[preset_name], height=150, key="sys_prompt_area")
    user_text = st.text_area("User text (your data or example)", height=150, placeholder="Paste a paragraph or dialogue here...", key="user_text_area")
    run = st.button("Run", type="primary", key="run_playground_btn")

    if run and user_text.strip():
        llm = get_llm(provider, temperature, user_key, local_model)
        with st.spinner("Generating..."):
            resp = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ])
        st.write(resp.content)


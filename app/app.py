import os, glob
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

MODEL_NAME = "llama3.2:3b-instruct"   # or "qwen2.5:3b-instruct"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

st.set_page_config(page_title="Political Narratives — Local Q&A & Playground", layout="wide")
st.title("Political Narratives — Local Q&A (RAG) + Prompt Playground")

with st.sidebar:
    st.markdown("### Local model (Ollama)")
    model_choice = st.selectbox("Ollama model", [MODEL_NAME, "qwen2.5:3b-instruct", "llama3.2:3b-instruct"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    st.caption("Tip: keep it low for factual answers.")

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

def load_files(patterns):
    paths = []
    for p in patterns:
        paths.extend(glob.glob(p, recursive=True))
    docs = []
    for path in paths:
        if os.path.isdir(path): continue
        if not path.endswith((".md", ".py", ".txt", ".yml", ".yaml", ".json")): continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            docs.append(Document(page_content=text, metadata={"source": os.path.relpath(path)}))
        except Exception as e:
            st.warning(f"Could not read {path}: {e}")
    return docs

@st.cache_resource(show_spinner=True)
def build_vectorstore():
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_docs = []

    # 1) Paper
    for d in load_pdf("data/paper.pdf"):
        for chunk in splitter.split_text(d.page_content):
            all_docs.append(Document(page_content=chunk, metadata=d.metadata))

    # 2) Repo docs (optional)
    repo_docs = load_files(["README.md", "examples/**/*.md", "*.md"])
    for d in repo_docs:
        for chunk in splitter.split_text(d.page_content):
            all_docs.append(Document(page_content=chunk, metadata=d.metadata))

    if not all_docs:
        return None

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(all_docs, embeddings)

vs = build_vectorstore()

tab1, tab2 = st.tabs(["Ask about the paper/repo (Local RAG)", "Prompt playground"])

with tab1:
    if vs is None:
        st.info("No documents indexed. Add a text-based PDF at data/paper.pdf or a README.md, then Rerun.")
    else:
        llm = ChatOllama(model=model_choice, temperature=temperature)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vs.as_retriever(search_kwargs={"k": 4}),
            chain_type="stuff",
            return_source_documents=True,
        )
        st.caption("Ask questions about the paper or this repository. Runs fully locally with Ollama.")
        q = st.text_input("Your question")
        if q:
            with st.spinner("Thinking locally..."):
                out = qa({"query": q})
            st.write(out["result"])
            if out.get("source_documents"):
                st.markdown("**Sources:**")
                for d in out["source_documents"]:
                    src = d.metadata.get("source", "unknown")
                    page = d.metadata.get("page", None)
                    st.code(f"{src}" + (f" (p.{page})" if page else ""))

with tab2:
    st.caption("Try prompts locally with Ollama. Choose a preset or write your own.")
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

    preset_name = st.selectbox("Preset", list(presets.keys()))
    system_prompt = st.text_area("System / Instruction", presets[preset_name], height=150)
    user_text = st.text_area("User text (your data or example)", height=150, placeholder="Paste a paragraph or dialogue here...")
    run = st.button("Run locally", type="primary")

    if run and user_text.strip():
        llm = ChatOllama(model=model_choice, temperature=temperature)
        with st.spinner("Generating locally..."):
            resp = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ])
        st.write(resp.content)


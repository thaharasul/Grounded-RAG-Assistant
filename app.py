
import hashlib
import os
from pathlib import Path

import chromadb
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -------------------- CONFIGURATION --------------------

ROOT = Path(__file__).resolve().parent
WIKI_DB = ROOT / "db"
UPLOAD_DB = ROOT / "uploaded_db"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

WIKI_NAME = "wikipedia"
UPLOAD_NAME = "uploaded_documents"

CHUNK_WORDS = 175
CHUNK_OVERLAP = 40
TOP_K = 3


# -------------------- STREAMLIT PAGE --------------------

st.set_page_config(
    page_title="Grounded RAG",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #0b1020;
        color: #e8edf8;
    }

    [data-testid="stSidebar"] {
        background: #10172a;
    }

    .hero {
        padding: 1.5rem;
        border: 1px solid #283653;
        border-radius: 18px;
        background: linear-gradient(135deg, #172440, #111827);
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        margin-bottom: 0.4rem;
        color: #f3f6ff;
    }

    .hero p {
        color: #b8c5dd;
        margin-bottom: 0;
    }

    .source-card {
        padding: 12px;
        margin: 8px 0;
        border: 1px solid #2b3955;
        border-radius: 10px;
        background: #111a2e;
    }

    div[data-testid="stChatMessage"] {
        border: 1px solid #26334d;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>📚 Grounded RAG Assistant</h1>
        <p>
            Ask questions about your documents or the indexed Wikipedia
            collection. Answers are generated using retrieved context.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------- CACHED RESOURCES --------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource
def get_wiki_client():
    if not WIKI_DB.exists():
        return None

    return chromadb.PersistentClient(path=str(WIKI_DB))


@st.cache_resource
def get_upload_client():
    UPLOAD_DB.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(UPLOAD_DB))


def get_collection(client, collection_name):
    try:
        return client.get_collection(name=collection_name)
    except Exception:
        return None


# -------------------- DOCUMENT PROCESSING --------------------

def extract_text(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".txt"):
        return uploaded_file.getvalue().decode(
            "utf-8",
            errors="replace",
        )

    if filename.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        pages = []

        for page in reader.pages:
            pages.append(page.extract_text() or "")

        return "\n".join(pages)

    raise ValueError("Only PDF and TXT files are supported.")


def chunk_text(text, chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()

    if not words:
        return []

    if overlap >= chunk_words:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    chunks = []
    step = chunk_words - overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_words])

        if chunk.strip():
            chunks.append(chunk)

        if start + chunk_words >= len(words):
            break

    return chunks


def index_uploaded_file(uploaded_file):
    text = extract_text(uploaded_file)

    if not text.strip():
        raise ValueError(
            "No extractable text found. If this is a scanned PDF, "
            "OCR may be required."
        )

    chunks = chunk_text(text)

    if not chunks:
        raise ValueError("No usable text chunks were created.")

    model = load_embedding_model()
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    client = get_upload_client()
    collection = client.get_or_create_collection(
        name=UPLOAD_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    file_hash = hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()[:16]

    # Stable IDs allow the same file to be indexed again safely.
    ids = [
        f"{file_hash}_{index}"
        for index in range(len(chunks))
    ]

    metadatas = [
        {
            "filename": uploaded_file.name,
            "chunk_index": index,
            "file_hash": file_hash,
        }
        for index in range(len(chunks))
    ]

    # Remove an earlier version of this exact file before re-indexing.
    collection.delete(
        where={"file_hash": file_hash}
    )

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(chunks)


# -------------------- RETRIEVAL --------------------

def retrieve_documents(question, collection, top_k=TOP_K):
    if collection is None or collection.count() == 0:
        return []

    model = load_embedding_model()

    question_embedding = model.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    result = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    sources = []

    for index, document in enumerate(documents):
        metadata = (
            metadatas[index]
            if index < len(metadatas) and metadatas[index]
            else {}
        )

        distance = (
            distances[index]
            if index < len(distances)
            else None
        )

        sources.append(
            {
                "text": document,
                "metadata": metadata,
                "distance": distance,
            }
        )

    return sources


def retrieve_from_wikipedia(question):
    client = get_wiki_client()

    if client is None:
        return []

    collection = get_collection(client, WIKI_NAME)

    return retrieve_documents(question, collection)


def retrieve_from_uploads(question):
    client = get_upload_client()
    collection = get_collection(client, UPLOAD_NAME)

    return retrieve_documents(question, collection)


# -------------------- GROQ ANSWER GENERATION --------------------

def generate_answer(question, sources):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Set it in your terminal "
            "before starting Streamlit."
        )

    if not sources:
        return (
            "I couldn't find relevant information in the selected "
            "knowledge base. Try a different question or upload "
            "a document containing the answer."
        )

    context_parts = []

    for index, source in enumerate(sources, start=1):
        filename = source["metadata"].get(
            "filename",
            "Wikipedia collection",
        )

        context_parts.append(
            f"[Source {index}: {filename}]\n{source['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a helpful Retrieval-Augmented Generation (RAG) assistant.

Answer the user's question using ONLY the retrieved context below.

Rules:
1. Do not invent facts that are not supported by the context.
2. If the context does not contain the answer, say so clearly.
3. Explain the answer in simple, direct language.
4. Cite supporting passages using [Source 1], [Source 2], etc.
5. Distinguish facts explicitly stated in the context from conclusions
   that are reasonably inferred.
6. Do not claim to have read anything outside the supplied context.

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    client = Groq(api_key=api_key)
    last_error = None

    for model_name in LLM_MODELS:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.2,
                max_completion_tokens=1024,
            )

            answer = response.choices[0].message.content

            if answer and answer.strip():
                return answer.strip()

            last_error = RuntimeError(
                f"{model_name} returned an empty answer."
            )

        except Exception as error:
            last_error = error
            print(f"Groq model {model_name} failed: {error}")

    raise RuntimeError(
        f"Answer generation failed. Last error: {last_error}"
    ) from last_error


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.title("⚙️ Settings")

    knowledge_source = st.radio(
        "Knowledge base",
        [
            "Uploaded documents",
            "Wikipedia collection",
        ],
    )

    top_k = st.slider(
        "Retrieved passages",
        min_value=1,
        max_value=6,
        value=TOP_K,
    )

    st.divider()

    st.subheader("📄 Upload documents")

    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button(
        "Index uploaded documents",
        use_container_width=True,
    ):
        if not uploaded_files:
            st.warning("Upload at least one PDF or TXT file first.")
        else:
            total_chunks = 0

            progress = st.progress(0)

            for index, file in enumerate(uploaded_files):
                try:
                    with st.spinner(f"Indexing {file.name}..."):
                        count = index_uploaded_file(file)

                    total_chunks += count
                    st.success(
                        f"{file.name}: indexed {count} chunks."
                    )

                except Exception as error:
                    st.error(f"{file.name}: {error}")

                progress.progress(
                    (index + 1) / len(uploaded_files)
                )

            st.info(
                f"Indexing finished. {total_chunks} chunks processed."
            )

    st.divider()

    st.caption("Answer generation: Groq API")
    st.caption("Embeddings: Sentence Transformers")
    st.caption("Vector database: ChromaDB")


# -------------------- CHAT INTERFACE --------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View retrieved sources"):
                for index, source in enumerate(
                    message["sources"],
                    start=1,
                ):
                    filename = source["metadata"].get(
                        "filename",
                        "Wikipedia collection",
                    )

                    st.markdown(
                        f"**Source {index}: {filename}**"
                    )
                    st.write(source["text"])

                    if source.get("distance") is not None:
                        st.caption(
                            "Cosine distance: "
                            f"{source['distance']:.4f}"
                        )

question = st.chat_input("Ask a question about your knowledge base...")

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your knowledge base..."):
                if knowledge_source == "Uploaded documents":
                    sources = retrieve_documents(
                        question,
                        get_collection(
                            get_upload_client(),
                            UPLOAD_NAME,
                        ),
                        top_k=top_k,
                    )
                else:
                    sources = retrieve_documents(
                        question,
                        get_collection(
                            get_wiki_client(),
                            WIKI_NAME,
                        ),
                        top_k=top_k,
                    )

            if not sources:
                answer = (
                    "I couldn't find any indexed passages in this "
                    "knowledge base. Upload and index documents, or "
                    "check that your Wikipedia database exists and "
                    "contains the 'wikipedia' collection."
                )
            else:
                with st.spinner("Generating an answer with Groq..."):
                    answer = generate_answer(question, sources)

            st.markdown(answer)

            if sources:
                with st.expander("View retrieved sources"):
                    for index, source in enumerate(
                        sources,
                        start=1,
                    ):
                        filename = source["metadata"].get(
                            "filename",
                            "Wikipedia collection",
                        )

                        st.markdown(
                            f"**Source {index}: {filename}**"
                        )
                        st.write(source["text"])

                        if source.get("distance") is not None:
                            st.caption(
                                "Cosine distance: "
                                f"{source['distance']:.4f}"
                            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )

        except Exception as error:
            error_message = f"Something went wrong: {error}"
            st.error(error_message)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )


# -------------------- FOOTER --------------------

st.divider()

st.caption(
    "RAG retrieves relevant passages before generating answers. "
    "Always verify important information against the original sources."
)

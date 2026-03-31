import streamlit as st
from dotenv import load_dotenv
import tempfile

from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", layout="centered")

st.title("📄 Multi-PDF RAG Chatbot")
st.caption("Upload PDFs and ask questions")

# ------------------ RESET BUTTON ------------------
if st.button("🔄 Reset Chat"):
    st.session_state.messages = []

# ------------------ FILE UPLOAD ------------------
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# ------------------ LOAD MODELS ------------------
@st.cache_resource
def load_llm():
    return ChatMistralAI(model="mistral-small-latest")

llm = load_llm()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer only from given context. If not found, say I don't know."),
    ("human", """
    Chat History:
    {history}

    Context:
    {context}

    Question:
    {question}
    """)
])

# ------------------ PROCESS PDFs ------------------
if uploaded_files:

    all_chunks = []

    with st.spinner("Processing PDFs..."):
        for file in uploaded_files:

            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                temp_path = tmp.name

            # Load PDF
            loader = PyPDFLoader(temp_path)
            docs = loader.load()

            # Add metadata (IMPORTANT)
            for doc in docs:
                doc.metadata["source"] = file.name

            # Split
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            chunks = splitter.split_documents(docs)

            all_chunks.extend(chunks)

    # Create vectorstore
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_model
    )

    # 🔥 Safe MMR (no fetch_k error)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3}
    )

    st.success("✅ PDFs processed! Ask your question.")

    # ------------------ CHAT ------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask something...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # Retrieve docs
        docs = retriever.invoke(user_input)
        context = "\n".join([doc.page_content for doc in docs])

        # Prompt
        history = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in st.session_state.messages[-5:]   # last 5 messages
        ])

        final_prompt = prompt.invoke({
            "context": context,
            "question": user_input,
            "history": history
        })

        # LLM
        response = llm.invoke(final_prompt)
        answer = response.content

        # Show answer
        with st.chat_message("assistant"):
            st.markdown(answer)

            # 🔥 Show sources
            with st.expander("📚 Sources"):
                for doc in docs:
                    st.write(f"📄 {doc.metadata.get('source')}")

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("📌 Upload at least one PDF to start.")
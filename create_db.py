from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


data = PyPDFLoader(r"C:\Users\HP\Desktop\RAG\document_loaders\java_notes.pdf")

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

chunks = splitter.split_documents(documents=docs)

print(len(chunks))

embedding_model = HuggingFaceEmbeddings()

vectorstore = Chroma.from_documents(
    documents = chunks,
    persist_directory= "chroma_db",
    embedding=embedding_model
)
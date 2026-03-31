# splitting
# embeddings
# vecotordb
# retriving
# generating
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


embedding_model = HuggingFaceEmbeddings()

vectorstore = Chroma(
    persist_directory= "chroma_db",
    embedding_function = embedding_model
)

retriver = vectorstore.as_retriever(
    search_type = 'mmr',
    search_kwargs = {
        'k' : 3,
        'fetch_k' : 10,
        'lambda_mult' : 0.5
    }
)

llm = ChatMistralAI(model = "mistral-small-latest")

prompt = ChatPromptTemplate.from_messages([
    ('system',"""You are an AI agent,
     User provides you a context and give answer to question only according to context.
     If You dont know tell i dont know the answer"""),
    ("human", """
    Chat History:
    {history}
    Context:
    {context}
    Question:
    {question}
    """)
])

chat_history = []

print("RAG SYSTEM CREATED")

print("ENTER 0 TO EXIT")

while True:
    query = input("You : ")
    if query == '0':
        break
    docs = retriver.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    # convert history to text
    history_text = "\n".join(chat_history[-5:])  # last 5 messages

    final_prompt = prompt.invoke({
        'context': context,
        'question': query,
        'history': history_text
    })
    response = llm.invoke(final_prompt)
    chat_history.append(f"user: {query}")
    chat_history.append(f"assistant: {response.content}")
    print("Bot : ",response.content)

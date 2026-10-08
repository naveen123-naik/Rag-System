#Document loader
from langchain_community.document_loaders import PyPDFLoader
file_path = "https://nationalinsurance.nic.co.in/sites/default/files/National%20Mediclaim%20Policy.pdf"

loader = PyPDFLoader(file_path)
doc = loader.load()

#split

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size = 1000,
  chunk_overlap = 200
)
all_split = text_splitter.split_documents(doc)

#Embedding

from langchain_huggingface import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(
  model_name = 'sentence-transformers/all-mpnet-base-v2'
)

#vector Store
from langchain_chroma import Chroma
vector_store = Chroma(
  collection_name='policy_collection',
  embedding_function=embedding_model,
  persist_directory='./chroma_langchain_db'
)

document_id = vector_store.add_documents(documents=all_split)
sample = vector_store.get(limit=1, include=["embeddings", "documents"])

def retrieve_context(query: str, k:int = 2):
  retrieved_docs = vector_store.similarity_search(query, k)
  docs_content = ""
  for doc in retrieved_docs:
    docs_content += f"Source: {doc.metadata}"
    docs_content += f"Content: {doc.page_content}"
  return docs_content, retrieved_docs

from langchain.chat_models import init_chat_model
from os import environ
from dotenv import load_dotenv
load_dotenv()

gemini_api_key = environ.get('GEMINI_API_KEY')

model = init_chat_model(
  "google_genai:gemini-2.5-flash",
  api_key = gemini_api_key
)


def docu_chat(user_query):
  context, source_docs = retrieve_context(user_query, k=2)
  system_message = f"""You are a helpful chatbot.
                     Use only the following pieces of context to answer the 
                     question. Don't makeup any new information: {context} """

  messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_query}
  ]
  response = model.invoke(messages)
  return {
  "answer": response.content,
  "source_documents": source_docs,
  "context_used": context
}
result = docu_chat("What is the coverage duration for pre-hospitalization and post-hospitalization medical expenses?")

print(result["answer"])
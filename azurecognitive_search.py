import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_community.document_loaders import AzureBlobStorageContainerLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import nltk

nltk.download('punkt_tab')

load_dotenv()

vector_store_address: str = f"https://{os.environ.get('AZURE_AI_SEARCH_SERVICE_NAME')}.search.windows.net"

embeddings: OpenAIEmbeddings = OpenAIEmbeddings()
index_name: str = "rag-app-vectordb"
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key = os.environ.get("AZURE_AI_SEARCH_API_KEY"),
    index_name= index_name,
    embedding_function=embeddings.embed_query,
)

loader = AzureBlobStorageContainerLoader(
    conn_str=os.environ.get("AZURE_CONN_STRING"),
    container = os.environ.get("CONTAINER_NAME"),
)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size = 150, chunk_overlap = 20)
docs = text_splitter.split_documents(documents)
vector_store.add_documents(documents=docs)

print("Data has been loaded into vectorstore successfully")

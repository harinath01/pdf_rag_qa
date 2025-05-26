from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant

def get_vector_store():
    client = QdrantClient(url="http://localhost:6333")
    embeddings = OpenAIEmbeddings()
    
    return Qdrant(
        client=client,
        collection_name="pdf_collection",
        embeddings=embeddings,
    ) 
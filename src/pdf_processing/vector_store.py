from enum import Enum
from typing import Optional
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_weaviate import WeaviateVectorStore
import weaviate

class VectorDB(Enum):
    QDRANT = "qdrant"
    WEAVIATE = "weaviate"

def get_vector_store(vdb: VectorDB = VectorDB.QDRANT) -> Qdrant | WeaviateVectorStore:
    embeddings = OpenAIEmbeddings()
    
    if vdb == VectorDB.QDRANT:
        client = QdrantClient(url="http://localhost:6333")
        return Qdrant(
            client=client,
            collection_name="pdf_collection",
            embeddings=embeddings,
        )
    elif vdb == VectorDB.WEAVIATE:
        client = weaviate.connect_to_local()
        return WeaviateVectorStore(
            client=client,
            index_name="pdf_collection",
            text_key="content",
            attributes=["pdf_id", "chunk_id", "type", "bbox", "title", "parent_title"],
            embedding=embeddings,
        )
    else:
        raise ValueError(f"Unsupported vector database: {vdb}") 
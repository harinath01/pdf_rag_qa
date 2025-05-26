from langchain.schema import Document
from langchain_community.vectorstores import Qdrant
from pdf_processing.chunker import Chunk
from typing import List

from src.pdf_processing.vector_store import get_vector_store

def create_langchain_documents(chunks: List[Chunk]) -> List[Document]:
    def create_citation_dict(citation) -> dict:
        return {
            "page": citation.page,
            "bbox": [citation.bbox.x0, citation.bbox.y0, citation.bbox.x1, citation.bbox.y1]
        }
    
    def create_title_dict(title) -> dict:
        return {
            "text": title.text,
            "page": title.page,
            "bbox": [title.bbox.x0, title.bbox.y0, title.bbox.x1, title.bbox.y1]
        }
    
    def create_metadata(chunk: Chunk) -> dict:
        metadata = {
            "chunk_id": chunk.chunk_id,
            "type": chunk.type,
            "citations": [create_citation_dict(cit) for cit in chunk.content.citations]
        }
        
        if chunk.title:
            metadata["title"] = create_title_dict(chunk.title)
        
        if chunk.parent_title:
            metadata["parent_title"] = create_title_dict(chunk.parent_title)
            
        return metadata
    
    return [
        Document(
            page_content=chunk.get_content(),
            metadata=create_metadata(chunk)
        )
        for chunk in chunks
    ]



def store_chunks_in_vector_store(chunks: List[Chunk], collection_name: str = "pdf_collection") -> Qdrant:    
    documents = create_langchain_documents(chunks)
    vector_store = get_vector_store()
    vector_store.add_documents(documents)
    return vector_store
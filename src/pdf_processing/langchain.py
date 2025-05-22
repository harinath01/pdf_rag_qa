from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from pdf_processing.chunker import Chunk
from langchain_openai import OpenAIEmbeddings
from typing import List

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


def create_vector_store(chunks: List[Chunk], save_path: str = "vector_store") -> FAISS:    
    documents = create_langchain_documents(chunks)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(save_path)
    
    return vector_store
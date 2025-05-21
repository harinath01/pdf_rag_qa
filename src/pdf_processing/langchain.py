from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from pdf_processing.chunker import Chunk
from langchain_openai import OpenAIEmbeddings
from typing import List

def create_langchain_documents(chunks: List[Chunk]) -> List[Document]:
    documents = []
    for chunk in chunks:
        metadata = {
            "chunk_id": chunk.chunk_id,
            "type": chunk.type,
            "citations": [{"page": cit.page, "bbox": [cit.bbox.x0, cit.bbox.y0, cit.bbox.x1, cit.bbox.y1]} for cit in chunk.content.citations]
        }
        
        if chunk.title:
            metadata["title"] = {
                "text": chunk.title.text,
                "page": chunk.title.page,
                "bbox": [chunk.title.bbox.x0, chunk.title.bbox.y0, chunk.title.bbox.x1, chunk.title.bbox.y1]
            }
        
        if chunk.parent_title:
            metadata["parent_title"] = {
                "text": chunk.parent_title.text,
                "page": chunk.parent_title.page,
                "bbox": [chunk.parent_title.bbox.x0, chunk.parent_title.bbox.y0, chunk.parent_title.bbox.x1, chunk.parent_title.bbox.y1]
            }
        
        doc = Document(
            page_content=chunk.get_content(),
            metadata=metadata
        )
        documents.append(doc)
    
    return documents


def create_vector_store(chunks: List[Chunk], save_path: str = "vector_store") -> FAISS:    
    documents = create_langchain_documents(chunks)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(save_path)
    
    return vector_store
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
            "citations": [{"page": cit.page, "bbox": cit.bbox.model_dump()} for cit in chunk.content.citations]
        }
        
        if chunk.title:
            metadata["title"] = chunk.title.model_dump()
        
        if chunk.parent_title:
            metadata["parent_title"] = chunk.parent_title.model_dump()
        
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
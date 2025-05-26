import argparse
from dotenv import load_dotenv
import uuid
from typing import Optional
from pdf_processing.parser import parse_pdf_to_json
from pdf_processing.chunker import chunk_json_output
from pdf_processing.vector_store import get_vector_store, VectorDB
from langchain.schema import Document

def create_langchain_documents(chunks, pdf_id: str):
    """Create LangChain documents with pdf_id in metadata"""
    documents = []
    for chunk in chunks:
        metadata = {
            "chunk_id": chunk.chunk_id,
            "type": chunk.type,
            "pdf_id": pdf_id,
            "citations": [{
                "page": citation.page,
                "bbox": [citation.bbox.x0, citation.bbox.y0, citation.bbox.x1, citation.bbox.y1]
            } for citation in chunk.content.citations]
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
            
        documents.append(Document(
            page_content=chunk.get_content(),
            metadata=metadata
        ))
    
    return documents

def process_pdf(pdf_path: str, vdb: VectorDB = VectorDB.QDRANT, page_range: Optional[str] = None) -> str:
    pdf_id = str(uuid.uuid4())
    
    result = parse_pdf_to_json(pdf_path, page_range=page_range)
    chunks = chunk_json_output(result)
    documents = create_langchain_documents(chunks, pdf_id)
    
    vector_store = get_vector_store(vdb)
    vector_store.add_documents(documents)
    
    return pdf_id

if __name__ == "__main__":
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Process a PDF file and store it in vector database')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--vdb', type=str, choices=[vdb.value for vdb in VectorDB], 
                       default=VectorDB.QDRANT.value, help='Vector database to use')
    parser.add_argument('--page_range', type=str, default=None, help='Page range to process (e.g., "1-10")')
    
    args = parser.parse_args()
    
    pdf_id = process_pdf(args.pdf_path, VectorDB(args.vdb), args.page_range)
    
    print(f"\n✅ PDF processed successfully!")
    print(f"📚 PDF ID: {pdf_id}")
    print(f"💾 Vector DB: {args.vdb}")
    print("\nUse this ID to ask questions about the document:")
    print(f"python ask_question.py {pdf_id} --vdb {args.vdb}") 
import json
import argparse
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from qdrant_client.models import Filter, FieldCondition, MatchValue
from pdf_processing.vector_store import get_vector_store, VectorDB
from weaviate.classes.query import Filter as WeaviateFilter

def ask_question(pdf_id: str, question: str, vdb: VectorDB = VectorDB.QDRANT, top_k: int = 1) -> dict:
    vector_store = get_vector_store(vdb)
    
    if vdb == VectorDB.QDRANT:
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="metadata.pdf_id",
                    match=MatchValue(value=pdf_id)
                )
            ]
        )
        search_kwargs = {
            "k": top_k,
            "filter": filter_condition
        }
    else:
        search_kwargs = {
            "k": top_k,
            "filters": WeaviateFilter.by_property("pdf_id").equal(pdf_id)
        }
    
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        retriever=vector_store.as_retriever(search_kwargs=search_kwargs),
        return_source_documents=True
    )
    
    answer = qa.invoke({"query": question})
    
    return {
        "answer": answer["result"],
        "sources": answer["source_documents"]
    }

def format_source_document(doc):
    """Format a single source document for better readability."""
    metadata = doc.metadata
    output = []
    
    if "title" in metadata:
        output.append(f"📑 Section: {metadata['title']['text']}")
    
    output.append("\n📝 Content:")
    output.append(doc.page_content)
    
    output.append(f"\n🔍 Chunk ID: {metadata['chunk_id']}")
    
    return "\n".join(output)

if __name__ == "__main__":
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Ask questions about a processed PDF')
    parser.add_argument('pdf_id', help='PDF ID of the processed document')
    parser.add_argument('--vdb', type=str, choices=[vdb.value for vdb in VectorDB], 
                       default=VectorDB.QDRANT.value, help='Vector database to use')
    parser.add_argument('--top-k', type=int, default=1, help='Number of similar documents to retrieve')
    
    args = parser.parse_args()
    
    while True:
        print("\n" + "="*80)
        query = input("\n❓ Ask a question (or type 'exit'): ")
        if query.strip().lower() == "exit":
            break
            
        result = ask_question(args.pdf_id, query, VectorDB(args.vdb), args.top_k)
        
        print("\n" + "="*80)
        print("\n💡 Answer:")
        print(result['answer'])
        
        print("\n" + "="*80)
        print("\n📚 Sources:")
        for i, doc in enumerate(result['sources'], 1):
            print(f"\n[{i}] " + "-"*76)
            print(format_source_document(doc))
            
            highlights = []
            if "title" in doc.metadata:
                highlights.append({
                    "pageIndex": doc.metadata["title"]["page"],
                    "bbox": doc.metadata["title"]["bbox"],
                    "pageHeight": 792
                })
            
            for citation in doc.metadata["citations"]:
                highlights.append({
                    "pageIndex": citation["page"],
                    "bbox": citation["bbox"],
                    "pageHeight": 792
                })
            
            print("\n📍 Highlights:")
            print(json.dumps(highlights, indent=2)) 
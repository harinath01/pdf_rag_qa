import json
import argparse
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

def create_qa_chain(vector_store: FAISS, top_k=1):
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        retriever=vector_store.as_retriever(search_kwargs={"k": top_k}),
        return_source_documents=True
    )
    
    return qa

def format_source_document(doc):
    """Format a single source document for better readability."""
    metadata = doc.metadata
    output = []
    
    # Add title if present
    if "title" in metadata:
        output.append(f"📑 Section: {metadata['title']['text']}")
    
    # Add content
    output.append("\n📝 Content:")
    output.append(doc.page_content)
    
    # Add chunk ID
    output.append(f"\n🔍 Chunk ID: {metadata['chunk_id']}")
    
    return "\n".join(output)

if __name__ == "__main__":
    from pdf_processing.parser import parse_pdf_to_json
    from pdf_processing.chunker import chunk_json_output
    from pdf_processing.langchain import create_vector_store
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='PDF QA System')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--page_range', type=str, default=None, help='Page range to process (e.g., "1-10")')
    parser.add_argument('--top-k', type=int, default=1)
    args = parser.parse_args()
    
    load_dotenv()
    result = parse_pdf_to_json(args.pdf_path, page_range=args.page_range)
    chunks = chunk_json_output(result)
    vector_store = create_vector_store(chunks)
    qa = create_qa_chain(vector_store, args.top_k)
    
    while True:
        print("\n" + "="*80)
        query = input("\n❓ Ask a question (or type 'exit'): ")
        if query.strip().lower() == "exit":
            break
            
        answer = qa.invoke({"query": query})
        
        # Print answer
        print("\n" + "="*80)
        print("\n💡 Answer:")
        print(answer['result'])
        
        # Print sources
        print("\n" + "="*80)
        print("\n📚 Sources:")
        for i, doc in enumerate(answer["source_documents"], 1):
            print(f"\n[{i}] " + "-"*76)
            print(format_source_document(doc))
            
            # Print highlights in a more compact format
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
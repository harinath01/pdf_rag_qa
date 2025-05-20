import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv



def create_qa_chain(vector_store: FAISS):
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        retriever=vector_store.as_retriever(search_kwargs={"k": 1}),
        return_source_documents=True
    )
    
    return qa

if __name__ == "__main__":
    from pdf_processing.parser import parse_pdf_to_json
    from pdf_processing.chunker import chunk_json_output
    from pdf_processing.langchain import create_vector_store
    
    load_dotenv()
    result = parse_pdf_to_json("data/attention_is_all_you_need.pdf")
    chunks = chunk_json_output(result)
    vector_store = create_vector_store(chunks)
    qa = create_qa_chain(vector_store)
    
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.strip().lower() == "exit":
            break
            
        answer = qa.invoke({"query": query})
        print("\nAnswer:", answer['result'])
        
        print("\nSources:")
        for doc in answer["source_documents"]:
            metadata = doc.metadata
            print(doc.page_content)
            print(f"\nChunk {metadata['chunk_id']}:")
            if "title" in metadata:
                print(f"Section: {metadata['title']['text']}")
                print
            for citation in metadata["citations"]:
                print(f"- Page {citation['page']} (bbox: {citation['bbox']})")
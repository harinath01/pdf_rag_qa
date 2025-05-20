from pdf_processing.parser import parse_pdf_to_json
from pdf_processing.chunker import chunk_json_output


if __name__ == "__main__":
    result = parse_pdf_to_json("data/attention_is_all_you_need.pdf", page_range="9-10")
    chunks = chunk_json_output(result)
    
    for chunk in chunks[:20]:
        print(chunk)
        print("-"*100)
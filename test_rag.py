from src.rag_pipeline import RAGPipeline

def main():
    # Initialize the RAG pipeline
    pipeline = RAGPipeline()
    
    # Process a website
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    print(f"Processing website: {url}")
    num_docs = pipeline.process_website(url)
    print(f"Processed {num_docs} documents")
    
    # Test a query
    query = "What are the main applications of artificial intelligence?"
    print(f"\nQuery: {query}")
    answer = pipeline.query(query)
    print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


# Load environment variables
load_dotenv()


class RAGPipeline:
    """
    RAG pipeline using:
    - Firecrawl for scraping
    - Hugging Face embeddings (via LangChain)
    - Pinecone for vector DB
    - Groq for generation
    """

    def __init__(
        self,
        index_name="pavan",
        namespace="default",
        groq_api_key=None,
        pinecone_api_key=None,
        pinecone_environment=None,
        firecrawl_api_key=None,
    ):
        print("🔧 Initializing RAG pipeline...")

        # --- Lazy imports to avoid circular dependency ---
        from src.scrapers.firecrawl_scraper import FirecrawlWebScraper
        from src.database.pinecone_db import PineconeDatabase
        from src.processors.groq_processor import GroqProcessor

        # --- Core components ---
        self.scraper = FirecrawlWebScraper(api_key=firecrawl_api_key)
        self.db = PineconeDatabase(
            index_name=index_name,
            namespace=namespace,
            api_key=pinecone_api_key,
            environment=pinecone_environment,
        )
        self.processor = GroqProcessor(api_key=groq_api_key)

        # --- Hugging Face embeddings (LangChain native) ---
        model_name = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        print(f"🧠 Using Hugging Face embeddings: {model_name}")
        self.embedder = HuggingFaceEmbeddings(model_name=model_name)

        # --- Pinecone vector store ---
        self.vector_store = self.db.create_vector_store(self.embedder)
        if not self.vector_store:
            raise RuntimeError("❌ Failed to initialize Pinecone vector store.")
        print(f"✅ Connected to Pinecone index: {index_name}")

        # --- Text splitter ---
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

        print("✅ RAG pipeline initialized successfully!")

    # ------------------------------------------------------------------
    def process_website(self, url, mode="scrape"):
        """Scrape or crawl a website, split, embed, and store in Pinecone."""
        print(f"🌐 Processing {url} in {mode.upper()} mode...")

        # Crawl or scrape
        documents = (
            self.scraper.crawl_website(url)
            if mode == "crawl"
            else self.scraper.scrape_website(url)
        )

        if not documents:
            print("⚠️ No documents retrieved.")
            return 0, "No content."

        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        print(f"🧩 Split {len(documents)} docs into {len(chunks)} chunks")

        # Add to Pinecone
        total_added = 0
        for i in range(0, len(chunks), 50):
            batch = chunks[i:i+50]
            try:
                ids = self.db.add_documents(self.vector_store, batch)
                total_added += len(ids)
                print(f"✅ Added batch {i//50 + 1}: {len(ids)} docs")
            except Exception as e:
                print(f"❌ Error adding batch {i//50 + 1}: {e}")

        # Summarize
        summary = self.generate_content_summary(documents)
        return total_added, summary

    # ------------------------------------------------------------------
    def generate_content_summary(self, documents):
        """Summarize using Groq LLM."""
        try:
            combined = "\n\n".join([d.page_content for d in documents[:5]])[:12000]
            prompt = f"""
Summarize the following web content concisely.

CONTENT:
{combined}
"""
            summary = self.processor.generate_text(prompt)
            print("🧠 Summary generated successfully!")
            return summary
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            return "Summary generation failed."

    # ------------------------------------------------------------------
    def query(self, query_text, k=4):
        """Query Pinecone and generate answer via Groq."""
        try:
            print(f"🔎 Querying knowledge base for: {query_text}")
            results = self.db.similarity_search(self.vector_store, query_text, k=k)
            if not results:
                return "No relevant info found."

            context = "\n\n".join([d.page_content for d in results])[:12000]
            prompt = f"""
Use ONLY the following context to answer:

CONTEXT:
{context}

QUESTION:
{query_text}

RULES:
- Use only facts from context
- If info missing, say "I don't have enough information."
"""
            answer = self.processor.generate_text(prompt)
            print("✅ Query answered successfully!")
            return answer
        except Exception as e:
            print(f"❌ Error answering query: {e}")
            return "Query processing failed."

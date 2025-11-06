import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PineconeDatabase:
    """
    A class to handle Pinecone vector database operations.
    """

    def __init__(self, index_name="pavan", namespace="default", api_key=None, environment=None):
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.environment = environment or os.getenv("PINECONE_ENVIRONMENT")

        if not self.api_key:
            raise ValueError("Pinecone API key not provided and not found in environment variables.")
        if not self.environment:
            raise ValueError("Pinecone environment not provided and not found in environment variables.")

        self.index_name = index_name
        self.namespace = namespace

    # ------------------------------------------------------------------
    def create_vector_store(self, embedding_function):
        try:
            from pinecone import Pinecone, ServerlessSpec
            from langchain_pinecone import PineconeVectorStore

            print("🔧 [DEBUG] Initializing Pinecone (v3 SDK)...")
            pc = Pinecone(api_key=self.api_key)
            existing_indexes = [index["name"] for index in pc.list_indexes()]
            print(f"📋 Existing Pinecone indexes: {existing_indexes}")

            if self.index_name not in existing_indexes:
                print(f"🆕 Index '{self.index_name}' does not exist. Creating it now...")
                embedding_dim = int(os.getenv("EMBEDDING_DIMENSION", "768"))
                pc.create_index(
                    name=self.index_name,
                    dimension=embedding_dim,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=self.environment)
                )
                print(f"✅ Index '{self.index_name}' created successfully.")
            else:
                print(f"✅ Using existing index '{self.index_name}'.")

            vector_store = PineconeVectorStore.from_existing_index(
                index_name=self.index_name,
                embedding=embedding_function,
                namespace=self.namespace
            )
            print(f"✅ Connected to Pinecone index '{self.index_name}' under namespace '{self.namespace}'.")
            return vector_store

        except Exception as e:
            print(f"❌ Error creating vector store: {str(e)}")
            return None

    # ------------------------------------------------------------------
    def add_documents(self, vector_store, documents):
        """Add documents safely with metadata cleaning."""
        try:
            if not vector_store:
                raise ValueError("Vector store not initialized before adding documents.")

            print(f"📥 Adding {len(documents)} documents to Pinecone...")

            # 🧹 Clean metadata
            for doc in documents:
                if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                    clean_meta = {}
                    for k, v in doc.metadata.items():
                        if v is None:
                            continue
                        elif isinstance(v, (str, int, float, bool)):
                            clean_meta[k] = v
                        elif isinstance(v, list):
                            clean_meta[k] = [str(x) for x in v]
                        else:
                            clean_meta[k] = str(v)
                    doc.metadata = clean_meta

            ids = vector_store.add_documents(documents)
            print(f"✅ Successfully added {len(ids)} documents to Pinecone.")
            return ids

        except Exception as e:
            print(f"❌ Error adding documents to vector store: {str(e)}")
            return []

    # ------------------------------------------------------------------
    def similarity_search(self, vector_store, query, k=4):
        """Perform a similarity search."""
        try:
            if not vector_store:
                raise ValueError("Vector store not initialized before performing similarity search.")
            print(f"🔍 Performing similarity search for query: {query[:80]}...")
            results = vector_store.similarity_search(query, k=k)
            print(f"✅ Retrieved {len(results)} similar documents.")
            return results
        except Exception as e:
            print(f"❌ Error performing similarity search: {str(e)}")
            return []

import os
from firecrawl import FirecrawlApp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class FirecrawlWebScraper:
    """Class for scraping or crawling webpages using the Firecrawl API."""

    def __init__(self, api_key=None):
        """
        Initialize the Firecrawl web scraper with the specified API key.

        Args:
            api_key (str, optional): The Firecrawl API key.
                                     If not provided, it will be loaded from the environment.
        """
        self.api_key = api_key or os.getenv("FIRECRAWL_API_KEY")

        if not self.api_key:
            raise ValueError(
                "❌ Firecrawl API key not found. Please set FIRECRAWL_API_KEY in your .env or pass it explicitly."
            )

        try:
            self.client = FirecrawlApp(api_key=self.api_key)
            print("✅ Firecrawl client initialized successfully.")
        except Exception as e:
            raise ValueError(f"❌ Error initializing Firecrawl scraper: {str(e)}")

    # ----------------------------------------------------------------------
    def scrape_url(self, url, mode="scrape", params=None):
        """
        Core function to call Firecrawl API for scraping or crawling.

        Args:
            url (str): Target webpage or site URL.
            mode (str): Either "scrape" (single page) or "crawl" (entire site).
            params (dict, optional): Extra API parameters.

        Returns:
            list: List of LangChain Document objects.
        """
        try:
            from langchain_core.documents import Document

            scrape_params = params or {}
            documents = []

            # --- SCRAPE (single page) ---
            if mode == "scrape":
                response = self.client.scrape(url, **scrape_params)
                content = getattr(response, "markdown", None) or getattr(response, "html", None)
                metadata = getattr(response, "metadata", {}) or {}

                # Ensure metadata is a dictionary
                if not isinstance(metadata, dict):
                    if hasattr(metadata, "__dict__"):
                        metadata = metadata.__dict__
                    else:
                        metadata = {"metadata": str(metadata)}

                if content:
                    documents.append(Document(page_content=content, metadata=dict(metadata)))

            # --- CRAWL (entire website) ---
            elif mode == "crawl":
                response = self.client.crawl(url, **scrape_params)

                for item in getattr(response, "data", []):
                    # Handle both dict-style and object-style responses
                    if isinstance(item, dict):
                        content = (
                            item.get("markdown")
                            or item.get("html")
                            or item.get("rawHtml", "")
                        )
                        metadata = item.get("metadata", {})
                    else:
                        content = getattr(item, "markdown", None) or getattr(item, "html", None)
                        metadata = getattr(item, "metadata", {}) or {}

                        if not isinstance(metadata, dict) and hasattr(metadata, "__dict__"):
                            metadata = metadata.__dict__

                    if content:
                        documents.append(Document(page_content=content, metadata=dict(metadata)))

            else:
                raise ValueError(f"Unsupported mode: {mode}")

            print(f"✅ Scraped {len(documents)} documents from {url}")
            return documents

        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")
            return []

    # ----------------------------------------------------------------------
    def crawl_website(self, url, params=None):
        """
        Wrapper for full-site crawling.
        Example: Used when mode='crawl' in RAGPipeline.
        """
        return self.scrape_url(url, mode="crawl", params=params)

    # ----------------------------------------------------------------------
    def scrape_website(self, url, params=None):
        """
        Wrapper for single-page scraping.
        Example: Used when mode='scrape' in RAGPipeline.
        """
        return self.scrape_url(url, mode="scrape", params=params)

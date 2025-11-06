# Web Content Summarization and Query Agent

A Streamlit-based RAG (Retrieval Augmented Generation) application to process web content, store it in a vector database, and query it using natural language.

## Features

- **Web Scraping**: Process single URLs or crawl entire websites using Firecrawl
- **Content Processing**: Chunk and process web content for efficient storage and retrieval
- **Vector Storage**: Store embeddings in Pinecone for fast similarity search
- **Natural Language Querying**: Ask questions about processed content using OpenAI's language models
- **Content Summarization**: Automatically generate summaries of processed web content

## Prerequisites

You'll need API keys for the following services (the app also supports alternative providers):

- **OpenAI** - For embeddings and text generation ([Get API Key](https://platform.openai.com/api-keys))
- **Pinecone** - For vector storage and retrieval ([Sign Up](https://www.pinecone.io/))
- **Firecrawl** - For web scraping and crawling ([Get API Key](https://firecrawl.dev/))
- **Groq (optional)** - Alternative LLM provider. If you use Groq set `GROQ_API_KEY` and `GROQ_API_URL` in your `.env`. For embeddings you can set `GROQ_EMBEDDINGS_URL` or use local embeddings via `sentence-transformers`.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/Pavan-chegu/Webscraping-Agent.git
   cd webscraping-agent
   ```

2. Create and activate a virtual environment (Windows PowerShell shown):

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. Install dependencies. This project supports both LangChain 0.x (monolithic) and LangChain 1.x (modular). Choose one of the options below:

- Recommended (works with LangChain 0.x):

   ```powershell
   pip install -r requirements.txt
   # Or use the provided recommendations file:
   # pip install -r requirements-recommended.txt
   ```

- If you prefer LangChain 1.x (modular packages), install the modular packages instead of the monolithic `langchain` package:

  ```powershell
  pip install -r requirements.txt
  pip install langchain_text_splitters langchain_community langchain_openai
  # Optional for Groq support:
  pip install langchain_groq
  ```

4. (Optional) If `firecrawl-py` is not pulled in by `requirements.txt`, install it explicitly:

   ```powershell
   pip install firecrawl-py
   ```

5. Create a `.env` file in the project root with your API keys (or provide them via the UI):

   ```ini
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=your_pinecone_environment
   FIRECRAWL_API_KEY=your_firecrawl_api_key
   # Optional Groq configuration
   GROQ_API_KEY=your_groq_api_key
   GROQ_API_URL=https://api.groq.com/v1/models/<model>/completions
   GROQ_EMBEDDINGS_URL=
   ```

## Usage

1. Start the Streamlit application:

   ```powershell
   python -m streamlit run streamlit_app.py
   ```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501).

3. Enter your API keys if not provided in the `.env` file.

4. Process a URL by entering it in the form and selecting the processing mode (`scrape` or `crawl`).

5. Ask questions about the processed content using the chat interface.

## Project Structure

- `streamlit_app.py`: Main Streamlit application
- `src/`: Source code directory
  - `rag_pipeline.py`: Main RAG pipeline implementation
  - `scrapers/`: Web scraping modules
  - `processors/`: Text processing modules
  - `database/`: Vector database modules



## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [LangChain](https://langchain.com/)
- [OpenAI](https://openai.com/)
- [Pinecone](https://www.pinecone.io/)
- [Firecrawl](https://firecrawl.dev/)
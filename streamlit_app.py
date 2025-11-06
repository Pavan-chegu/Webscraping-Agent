import streamlit as st
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rag_pipeline import RAGPipeline

# Set page configuration
st.set_page_config(
    page_title="Webscrape RAG Agent",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state for API keys
if "api_keys_submitted" not in st.session_state:
    st.session_state.api_keys_submitted = False

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
    st.session_state.groq_api_key = ""

if "pinecone_api_key" not in st.session_state:
    st.session_state.pinecone_api_key = ""

if "pinecone_environment" not in st.session_state:
    st.session_state.pinecone_environment = ""

if "firecrawl_api_key" not in st.session_state:
    st.session_state.firecrawl_api_key = ""

# Add custom CSS for chat styling
st.markdown("""
<style>
/* Main layout improvements */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 0;
    max-width: 95%;
}

/* Make the chat area more visible and scrollable */
.main-chat-area {
    height: 60vh;
    overflow-y: auto;
    border: 1px solid #e6e6e6;
    border-radius: 0.5rem;
    padding: 1rem;
    margin-bottom: 1rem;
    background-color: #f9f9f9;
    display: block; /* Changed from flex to block */
    padding-bottom: 20px; /* Reduced space for fixed input */
}

/* Chat message styling */
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
}
.chat-message.user {
    background-color: #e6f7ff;
    color: #000000;
}
.chat-message.assistant {
    background-color: #f0f7ff;
    color: #000000;
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .content {
    width: 80%;
}

/* Make chat container scrollable */
.stChatFloatingInputContainer {
    position: relative !important;
    bottom: 0 !important;
    width: 100% !important;
    padding: 1rem !important;
    background-color: rgba(255, 255, 255, 0.95) !important;
    z-index: 100 !important;
}

/* Style for the fixed chat input container */
div.stChatInputContainer {
    position: sticky;
    bottom: 0;
    background-color: white;
    padding: 1rem 0;
    z-index: 100;
    box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
}

/* Make chat input more prominent */
div.stChatInputContainer input {
    border: 1px solid #e0e0e0;
    border-radius: 25px;
    padding: 10px 15px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}

/* Chat message container */
.stChatMessageContent {
    padding: 1rem !important;
    border-radius: 0.5rem !important;
}

/* User message styling */
.stChatMessageContent[data-testid="userChatMessage"] {
    background-color: #e6f7ff !important;
    color: #000000 !important;
}

/* Assistant message styling */
.stChatMessageContent[data-testid="assistantChatMessage"] {
    background-color: #f0f7ff !important;
    color: #000000 !important;
}

/* Notification styling */
.notification {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    background-color: #e6f7ff;
    color: #000000;
    border-left: 5px solid #5bc0de;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}

/* Improve sidebar appearance */
[data-testid="stSidebar"] {
    background-color: #f8f9fa;
    border-right: 1px solid #e9ecef;
}

/* Expandable sections */
.st-expander {
    background-color: #f8f9fa !important;
    border-radius: 0.5rem !important;
    margin-bottom: 0.5rem !important;
}

/* Make buttons more visible */
button[kind="primary"] {
    border-radius: 25px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

/* Ensure chat messages are visible */
div.stChatMessage {
    background-color: #f0f2f6 !important;
    border-radius: 0.5rem !important;
    margin-bottom: 0.8rem !important;
    padding: 0.8rem !important;
    border-left: 3px solid #4e8cff;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

div.stChatMessage[data-testid="chat-message-user"] {
    border-left: 3px solid #36b37e;
    background-color: #f0f7f0 !important;
}

div.stChatMessage[data-testid="chat-message-assistant"] {
    border-left: 3px solid #4e8cff;
    background-color: #f0f7ff !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed_urls" not in st.session_state:
    st.session_state.processed_urls = set()
    
# Function to save API keys to session state
def save_api_keys():
    # Save both OpenAI and Groq inputs; pipeline will prefer Groq if provided
    st.session_state.openai_api_key = st.session_state.openai_key_input if "openai_key_input" in st.session_state else ""
    st.session_state.groq_api_key = st.session_state.groq_key_input if "groq_key_input" in st.session_state else ""
    st.session_state.pinecone_api_key = st.session_state.pinecone_key_input
    st.session_state.pinecone_environment = st.session_state.pinecone_env_input
    st.session_state.firecrawl_api_key = st.session_state.firecrawl_key_input

    # Validate inputs before proceeding. Groq may be used as an alternative to OpenAI.
    if not (st.session_state.openai_api_key or st.session_state.groq_api_key):
        st.error("Either an OpenAI API key or a Groq API key is required.")
        return
    if not st.session_state.pinecone_api_key:
        st.error("Pinecone API key is required.")
        return
    if not st.session_state.pinecone_environment:
        st.error("Pinecone environment is required.")
        return
    if not st.session_state.firecrawl_api_key:
        st.error("Firecrawl API key is required.")
        return

    # Set keys as submitted and show a loading message
    st.session_state.api_keys_submitted = True

    # Initialize RAG pipeline with the provided API keys (Groq key preferred)
with st.spinner("Initializing RAG pipeline and validating API keys..."):
    try:
        print("Attempting to initialize RAG pipeline with API keys...")

        # Prefer Groq key if provided
        st.session_state.rag_pipeline = RAGPipeline(
            groq_api_key=st.session_state.groq_api_key or st.session_state.openai_api_key,
            pinecone_api_key=st.session_state.pinecone_api_key,
            pinecone_environment=st.session_state.pinecone_environment,
            firecrawl_api_key=st.session_state.firecrawl_api_key
        )

        print("RAG pipeline initialized successfully!")
        st.success("API keys validated and RAG pipeline initialized successfully!")
        st.session_state.api_keys_submitted = True

        print(
            f"Session state after initialization: api_keys_submitted={st.session_state.api_keys_submitted}, "
            f"rag_pipeline exists={hasattr(st.session_state, 'rag_pipeline')}"
        )

    except ValueError as ve:
        print(f"Validation Error: {str(ve)}")
        st.error(f"Validation Error: {str(ve)}")
        st.session_state.api_keys_submitted = False

    except Exception as e:
        print(f"Error initializing RAG pipeline: {str(e)}")
        st.error(f"Error initializing RAG pipeline: {str(e)}")
        st.session_state.api_keys_submitted = False
  
if "content_summaries" not in st.session_state:
    st.session_state.content_summaries = {}

# Function to process a URL
def process_url(url, mode):
    with st.spinner(f"Processing {url}..."):
        num_docs, summary = st.session_state.rag_pipeline.process_website(url, mode=mode)
        if num_docs > 0:
            st.session_state.processed_urls.add(url)
            st.session_state.content_summaries[url] = summary
            
            # Add a system message to chat history indicating content is ready
            ready_message = f"✅ I've processed {url} and extracted {num_docs} documents. I'm now ready to answer your questions about this content!"
            st.session_state.chat_history.append({"role": "assistant", "content": ready_message})
            
            return True, f"Successfully processed {num_docs} documents from {url}"
        else:
            return False, f"Failed to process {url}"

# Function to handle user queries
def handle_query(query):
    if not st.session_state.processed_urls:
        return "Please process at least one URL before asking questions."
    
    with st.spinner("Generating answer..."):
        answer = st.session_state.rag_pipeline.query(query)
        return answer

# Main app layout
st.title("🔍 Web Content Summarization and Query Agent")

# API Keys Input Form (shown only if keys haven't been submitted)
if not st.session_state.api_keys_submitted:
    st.markdown("""
    ## Welcome to the Web Content RAG Agent
    
    Before you can use this application, you need to provide your API keys for the following services:
    
    - **OpenAI API** - For embeddings and text generation ([Get API Key](https://platform.openai.com/api-keys))
    - **Pinecone** - For vector storage and retrieval ([Sign Up](https://www.pinecone.io/))
    - **Firecrawl** - For web scraping and crawling ([Get API Key](https://firecrawl.dev/))
    
    These keys will be stored in your browser's session and will not be saved on any server.
    """)
    
    # Create a form for API key inputs
    with st.form("api_keys_form"):
        st.text_input("OpenAI API Key", type="password", key="openai_key_input")
        st.text_input("Pinecone API Key", type="password", key="pinecone_key_input")
        st.text_input("Pinecone Environment", key="pinecone_env_input", 
                     help="e.g., 'us-west1-gcp', 'us-east1-aws', etc.")
        st.text_input("Firecrawl API Key", type="password", key="firecrawl_key_input")
        
        # Submit button
        submitted = st.form_submit_button("Save API Keys and Start")
        if submitted:
            save_api_keys()

# Debug message to help troubleshoot UI issues
if st.session_state.api_keys_submitted:
    if "rag_pipeline" not in st.session_state:
        st.error("API keys were validated but the RAG pipeline was not properly initialized. Please refresh the page and try again.")
        st.session_state.api_keys_submitted = False

# Main Application (shown only if API keys have been submitted and pipeline is initialized)
if st.session_state.api_keys_submitted and "rag_pipeline" in st.session_state:
    # Create two columns for the layout
    col1, col2 = st.columns([1, 2])
    
    # Sidebar for URL processing
    with col1:
        st.header("Process Web Content")
        
        url_input = st.text_input("Enter a URL to process:")
        mode = st.radio("Processing mode:", ["crawl", "scrape"], 
                       help="'crawl' processes all accessible subpages, 'scrape' processes only the given URL")
        
        if st.button("Process URL"):
            if url_input:
                success, message = process_url(url_input, mode)
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.warning("Please enter a URL")
        
        # Display processed URLs with expandable summaries
        if st.session_state.processed_urls:
            st.subheader("Processed Content:")
            
            # Add a notification about available content
            st.markdown("""
            <div class="notification">
                <p>✅ The following content has been processed and is available for questions:</p>
            </div>
            """, unsafe_allow_html=True)
            
            for url in st.session_state.processed_urls:
                with st.expander(f"📄 {url}"):
                    if url in st.session_state.content_summaries:
                        st.markdown(st.session_state.content_summaries[url])
                    else:
                        st.markdown("*No summary available*")
        
        # Add some information about the app
        st.markdown("---")
        st.markdown("""
        ### About this app
        
        This application uses:
        - **Firecrawl** for web scraping and crawling
        - **LangChain** for document processing and chunking
        - **Pinecone** for vector storage and retrieval
        - **OpenAI** for embeddings and text generation
        - **Streamlit** for the user interface
        
        #### How it works:
        1. Enter a URL and choose whether to scrape just that page or crawl the entire site
        2. The content is processed, chunked, and stored in a vector database
        3. Ask questions about the processed content
        4. The RAG agent will retrieve relevant information and provide accurate answers
        
        > **Note:** The agent only answers based on the content you've processed. It doesn't use external knowledge or make up information.
        """)
    
    # Chat interface
    with col2:
        st.header("Chat with the RAG Agent")
        
        # Add a welcome message if no chat history exists
        if not st.session_state.chat_history and not st.session_state.processed_urls:
            st.markdown("""
            <div class="notification">
            <h3>👋 Welcome to the Web Content RAG Agent!</h3>
            <p>To get started, please process a URL using the form on the left. Once content is processed, you can ask questions about it here.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Create a container for the scrollable chat area
    chat_container = st.container()
    
    # Wrap the chat messages in the scrollable container
    with chat_container:
        st.markdown('<div class="main-chat-area">', unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input - this will be fixed at the bottom due to CSS
    if query := st.chat_input("Ask a question about the processed content..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Generate and display assistant response
        with st.spinner("Thinking..."):
            response = handle_query(query)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
            
        # Force a rerun to update the UI immediately
        st.rerun()
import streamlit as st
import PyPDF2
import io
import datetime
import psutil
import platform
from typing import List, Dict, Any
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards
import traceback

# Page configuration
st.set_page_config(
    page_title="Medical Research Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical theme
def load_custom_css():
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-blue: #2E86AB;
        --secondary-blue: #A23B72;
        --light-blue: #F18F01;
        --background-light: #F8F9FA;
        --text-primary: #2C3E50;
        --success-green: #27AE60;
        --warning-orange: #F39C12;
        --error-red: #E74C3C;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        padding: 2rem 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: var(--background-light);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px;
        padding: 1rem 2rem;
        font-weight: 600;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        border-color: var(--primary-blue);
        transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-blue), var(--secondary-blue));
        color: white;
    }
    
    /* File upload area */
    .upload-area {
        border: 2px dashed var(--primary-blue);
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: rgba(46, 134, 171, 0.05);
        margin: 1rem 0;
    }
    
    /* Document cards */
    .doc-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid var(--primary-blue);
        transition: transform 0.2s ease;
    }
    
    .doc-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .doc-title {
        color: var(--primary-blue);
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .doc-metadata {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    
    .doc-preview {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        line-height: 1.4;
        color: #444;
    }
    
    /* Status indicators */
    .status-success {
        background-color: var(--success-green);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .status-warning {
        background-color: var(--warning-orange);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .status-error {
        background-color: var(--error-red);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    /* Query area */
    .query-area {
        background: linear-gradient(135deg, rgba(46, 134, 171, 0.1), rgba(162, 59, 114, 0.1));
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* History items */
    .history-item {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid var(--primary-blue);
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* System status */
    .system-status {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-blue), var(--secondary-blue));
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(46, 134, 171, 0.3);
    }
    
    /* Metrics styling */
    div[data-testid="metric-container"] {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid var(--primary-blue);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    }
    
    /* Alert boxes */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(135deg, var(--primary-blue), var(--secondary-blue));
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = []
    if 'document_texts' not in st.session_state:
        st.session_state.document_texts = {}
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'system_status' not in st.session_state:
        st.session_state.system_status = {
            'last_update': datetime.datetime.now(),
            'documents_processed': 0,
            'queries_made': 0,
            'total_pages': 0
        }

# PDF processing functions
def extract_text_from_pdf(pdf_file) -> Dict[str, Any]:
    """Extract text and metadata from PDF file"""
    try:
        # Create PDF reader
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Extract metadata
        metadata = {
            'filename': pdf_file.name,
            'page_count': len(pdf_reader.pages),
            'upload_time': datetime.datetime.now(),
            'file_size': pdf_file.size if hasattr(pdf_file, 'size') else 0
        }
        
        # Extract text from all pages
        full_text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            except Exception as e:
                st.warning(f"Could not extract text from page {page_num + 1}: {str(e)}")
                continue
        
        # Create preview (first 500 characters)
        preview_text = full_text[:500].strip()
        if len(full_text) > 500:
            preview_text += "..."
        
        return {
            'success': True,
            'metadata': metadata,
            'full_text': full_text,
            'preview': preview_text,
            'error': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'metadata': {'filename': pdf_file.name, 'error': str(e)},
            'full_text': '',
            'preview': '',
            'error': str(e)
        }

def get_system_info() -> Dict[str, Any]:
    """Get current system information"""
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            'cpu_usage': cpu_percent,
            'memory_total': memory.total / (1024**3),  # GB
            'memory_used': memory.used / (1024**3),    # GB
            'memory_percent': memory.percent,
            'platform': platform.system(),
            'processor': platform.processor(),
            'python_version': platform.python_version()
        }
    except:
        return {
            'cpu_usage': 0,
            'memory_total': 0,
            'memory_used': 0,
            'memory_percent': 0,
            'platform': 'Unknown',
            'processor': 'Unknown',
            'python_version': platform.python_version()
        }

# Tab content functions
def render_upload_tab():
    """Render the Upload tab content"""
    colored_header(
        label="📁 Document Upload",
        description="Upload PDF documents for medical research analysis",
        color_name="blue-70"
    )
    
    # File upload section
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Select one or more PDF files to upload. Maximum file size: 200MB per file."
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_files:
        st.subheader("📊 Processing Status")
        
        # Process uploaded files
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in [doc['metadata']['filename'] for doc in st.session_state.uploaded_documents]:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    # Extract text and metadata
                    result = extract_text_from_pdf(uploaded_file)
                    
                    if result['success']:
                        # Store in session state
                        st.session_state.uploaded_documents.append(result)
                        st.session_state.document_texts[uploaded_file.name] = result['full_text']
                        
                        # Update system status
                        st.session_state.system_status['documents_processed'] += 1
                        st.session_state.system_status['total_pages'] += result['metadata']['page_count']
                        st.session_state.system_status['last_update'] = datetime.datetime.now()
                        
                        st.success(f"✅ Successfully processed {uploaded_file.name}")
                    else:
                        st.error(f"❌ Failed to process {uploaded_file.name}: {result['error']}")
    
    # Display uploaded documents
    if st.session_state.uploaded_documents:
        st.subheader("📚 Uploaded Documents")
        
        for doc in st.session_state.uploaded_documents:
            metadata = doc['metadata']
            
            # Create document card
            st.markdown(f"""
            <div class="doc-card">
                <div class="doc-title">📄 {metadata['filename']}</div>
                <div class="doc-metadata">
                    <strong>Pages:</strong> {metadata['page_count']} | 
                    <strong>Uploaded:</strong> {metadata['upload_time'].strftime('%Y-%m-%d %H:%M:%S')} |
                    <strong>Size:</strong> {metadata.get('file_size', 0) / 1024:.1f} KB
                </div>
                <div class="doc-preview">
                    <strong>Preview:</strong><br>
                    {doc['preview']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.info("💡 No documents uploaded yet. Use the file uploader above to get started.")

def render_ask_tab():
    """Render the Ask tab content"""
    colored_header(
        label="🤔 Ask Questions",
        description="Query your uploaded documents using natural language",
        color_name="blue-70"
    )
    
    if not st.session_state.uploaded_documents:
        st.warning("⚠️ Please upload some documents first before asking questions.")
        return
    
    # Query input section
    st.markdown('<div class="query-area">', unsafe_allow_html=True)
    
    # Document selection
    doc_names = [doc['metadata']['filename'] for doc in st.session_state.uploaded_documents]
    selected_docs = st.multiselect(
        "Select documents to query (leave empty to search all)",
        doc_names,
        help="Choose specific documents to search, or leave empty to search all uploaded documents"
    )
    
    # Query input
    user_query = st.text_area(
        "Enter your question:",
        placeholder="e.g., What are the main findings about cardiovascular disease in these studies?",
        height=100,
        help="Ask questions about the content of your uploaded documents"
    )
    
    # Submit button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        submit_query = st.button("🔍 Submit Query", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if submit_query and user_query.strip():
        # Store query in history
        query_data = {
            'query': user_query,
            'timestamp': datetime.datetime.now(),
            'documents': selected_docs if selected_docs else doc_names,
            'status': 'processed'
        }
        st.session_state.query_history.append(query_data)
        st.session_state.system_status['queries_made'] += 1
        
        # Placeholder for RAG implementation
        st.subheader("🔍 Query Results")
        
        with st.spinner("Processing your query..."):
            import time
            time.sleep(2)  # Simulate processing time
            
            # Placeholder response
            st.success("Query processed successfully!")
            
            st.markdown("""
            <div class="doc-card">
                <div class="doc-title">🤖 AI Response</div>
                <div class="doc-metadata">
                    <span class="status-success">RAG Ready</span> - This is a placeholder for the RAG system implementation
                </div>
                <div class="doc-preview">
                    <strong>Your Query:</strong> {query}<br><br>
                    <strong>Response:</strong> This is where the RAG (Retrieval-Augmented Generation) system would provide 
                    intelligent answers based on your uploaded documents. The system would:
                    <br><br>
                    1. Search through the selected documents for relevant information<br>
                    2. Extract the most relevant passages<br>
                    3. Generate a comprehensive answer using AI<br>
                    4. Provide citations and references to specific documents<br><br>
                    <em>Integration with your preferred LLM (OpenAI, Anthropic, etc.) can be added here.</em>
                </div>
            </div>
            """.format(query=user_query), unsafe_allow_html=True)
    
    elif submit_query:
        st.error("Please enter a question before submitting.")

def render_history_tab():
    """Render the History tab content"""
    colored_header(
        label="📜 History",
        description="View your uploaded documents and query history",
        color_name="blue-70"
    )
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 Document History")
        
        if st.session_state.uploaded_documents:
            for i, doc in enumerate(st.session_state.uploaded_documents):
                metadata = doc['metadata']
                
                with st.expander(f"📄 {metadata['filename']}"):
                    st.write(f"**Pages:** {metadata['page_count']}")
                    st.write(f"**Upload Time:** {metadata['upload_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**File Size:** {metadata.get('file_size', 0) / 1024:.1f} KB")
                    
                    # Show preview
                    st.write("**Preview:**")
                    st.text(doc['preview'])
                    
                    # Delete button
                    if st.button(f"🗑️ Remove Document", key=f"delete_{i}"):
                        # Remove from session state
                        filename = metadata['filename']
                        st.session_state.uploaded_documents = [
                            d for d in st.session_state.uploaded_documents 
                            if d['metadata']['filename'] != filename
                        ]
                        if filename in st.session_state.document_texts:
                            del st.session_state.document_texts[filename]
                        
                        st.success(f"Removed {filename}")
                        st.rerun()
        else:
            st.info("No documents uploaded yet.")
    
    with col2:
        st.subheader("🤔 Query History")
        
        if st.session_state.query_history:
            for i, query in enumerate(reversed(st.session_state.query_history)):
                with st.expander(f"Query {len(st.session_state.query_history) - i}"):
                    st.write(f"**Question:** {query['query']}")
                    st.write(f"**Time:** {query['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**Documents:** {', '.join(query['documents'])}")
                    st.write(f"**Status:** {query['status']}")
        else:
            st.info("No queries made yet.")
    
    # Clear history buttons
    st.subheader("🧹 Cleanup")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear All Documents", type="secondary"):
            st.session_state.uploaded_documents = []
            st.session_state.document_texts = {}
            st.success("All documents cleared!")
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Query History", type="secondary"):
            st.session_state.query_history = []
            st.success("Query history cleared!")
            st.rerun()

def render_sidebar():
    """Render the sidebar with system information"""
    st.sidebar.markdown("""
    <div class="main-header">
        <h2>🏥 Medical Research Assistant</h2>
        <p>AI-Powered Document Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System status
    st.sidebar.subheader("🖥️ System Status")
    
    system_info = get_system_info()
    
    # System metrics
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.metric(
            "CPU Usage",
            f"{system_info['cpu_usage']:.1f}%",
            delta=None
        )
    
    with col2:
        st.metric(
            "Memory",
            f"{system_info['memory_used']:.1f}GB",
            delta=f"{system_info['memory_percent']:.1f}%"
        )
    
    # Application statistics
    st.sidebar.subheader("📊 Application Stats")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.metric(
            "Documents",
            st.session_state.system_status['documents_processed']
        )
    
    with col2:
        st.metric(
            "Queries",
            st.session_state.system_status['queries_made']
        )
    
    st.sidebar.metric(
        "Total Pages",
        st.session_state.system_status['total_pages']
    )
    
    # System information
    with st.sidebar.expander("💻 System Information"):
        st.write(f"**Platform:** {system_info['platform']}")
        st.write(f"**Python:** {system_info['python_version']}")
        st.write(f"**Total RAM:** {system_info['memory_total']:.1f} GB")
        st.write(f"**Processor:** {system_info['processor'][:50]}...")
    
    # Tips and help
    with st.sidebar.expander("💡 Tips & Help"):
        st.markdown("""
        **Getting Started:**
        1. Upload PDF documents in the Upload tab
        2. Ask questions about your documents in the Ask tab
        3. View your history in the History tab
        
        **Tips:**
        - Upload multiple PDFs at once
        - Use specific, detailed questions
        - Review document previews before querying
        - Check system resources regularly
        
        **Supported Files:**
        - PDF documents only
        - Maximum 200MB per file
        - Text-based PDFs work best
        """)

# Main application
def main():
    """Main application function"""
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Medical Research Assistant</h1>
        <p>Upload, analyze, and query medical documents with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📁 Upload", "🤔 Ask", "📜 History"])
    
    with tab1:
        render_upload_tab()
    
    with tab2:
        render_ask_tab()
    
    with tab3:
        render_history_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🏥 Medical Research Assistant | Built with Streamlit | 
        <span class="status-success">System Ready</span></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    # app.py

import os
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import chromadb
import logging
from ctransformers import AutoModelForCausalLM

# --- Assume these functions are in pdf_preprocessing.py ---
# To make this script runnable standalone for UI testing, we will create dummy functions.
# In a real scenario, you would use: from pdf_preprocessing import extract_text_robust, chunk_text, store_in_chroma
try:
    from pdf_preprocessing import extract_text_robust, chunk_text, store_in_chroma
    PDF_PREPROCESSING_AVAILABLE = True
except ImportError:
    PDF_PREPROCESSING_AVAILABLE = False
    logging.warning("pdf_preprocessing.py not found. Upload functionality will be disabled.")
    # Define dummy functions if the module is not found
    def extract_text_robust(pdf_file): return "This is dummy text from a PDF." * 100
    def chunk_text(text, chunk_size=800, overlap=150): return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]
    def store_in_chroma(chunks, filename): pass
# --- End of assumed functions ---


# --- Page Configuration ---
st.set_page_config(
    page_title="Medical PDF Analyzer",
    page_icon="🩺",
    layout="wide"
)

# --- Custom CSS for Medical Theme ---
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #F0F2F6;
    }
    /* Expander styling */
    .st-expander {
        border: 1px solid #0068C9;
        border-radius: 10px;
        background-color: #FFFFFF;
    }
    .st-expander header {
        font-weight: bold;
        color: #0068C9;
    }
    /* Main headers */
    h1, h2 {
        color: #1E3D59;
    }
    /* Buttons */
    .stButton>button {
        background-color: #0068C9;
        color: white;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #00509E;
        color: white;
    }
    /* Response box */
    .response-box {
        border: 1px solid #B0C4DE;
        border-radius: 8px;
        padding: 15px;
        background-color: #FFFFFF;
        margin-bottom: 20px;
    }
    .response-box p {
        font-family: 'sans serif';
        font-size: 16px;
        color: #333333;
    }
</style>
""", unsafe_allow_html=True)


# --- Caching for expensive operations ---
@st.cache_resource
def get_llm_model():
    """Load the LLM from the local file system."""
    model_path = os.path.join("models", "llama-2-7b-chat.Q4_K_M.gguf")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please follow the setup instructions.")
        return None
    try:
        llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type='llama',
            max_new_tokens=1024,
            context_length=3000,
            temperature=0.1,
            repetition_penalty=1.1
        )
        return llm
    except Exception as e:
        st.error(f"Failed to load the LLM. Error: {e}")
        return None

@st.cache_resource
def get_chroma_client():
    """Initialize a persistent ChromaDB client."""
    try:
        return chromadb.PersistentClient(path="./chroma_db")
    except Exception as e:
        st.error(f"Failed to initialize ChromaDB. Error: {e}")
        return None

# --- LLM and DB Initialization ---
llm = get_llm_model()
chroma_client = get_chroma_client()


# --- Session State Initialization ---
if "history" not in st.session_state:
    st.session_state.history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


# --- Core Logic Functions ---
def query_chroma(query_text: str, collection_name: str, n_results=4):
    """Queries the specified ChromaDB collection."""
    if not chroma_client:
        return None, "ChromaDB client is not available."
    try:
        collection = chroma_client.get_collection(name=collection_name)
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results, None
    except Exception as e:
        return None, f"Could not query collection '{collection_name}'. It might not exist yet. Error: {e}"

def generate_response(llm_instance, question: str, context_chunks: list):
    """Generates a response using the LLM with context."""
    if not llm_instance:
        return "LLM is not loaded. Cannot generate response."

    context_str = "\n\n".join([f"Source [{i+1}]:\n{chunk}" for i, chunk in enumerate(context_chunks)])

    prompt = f"""
    You are an expert medical assistant. Based SOLELY on the context provided below, answer the user's question.
    Your answer must be concise and directly derived from the text.
    When you use information from a source, you MUST cite it by including `[Source X]` at the end of the sentence, where X is the source number.
    If the context does not contain the answer, state that you cannot answer based on the provided documents.

    --- CONTEXT ---
    {context_str}

    --- QUESTION ---
    {question}

    --- ANSWER ---
    """
    try:
        response = llm_instance(prompt)
        return response
    except Exception as e:
        logging.error(f"Error during LLM inference: {e}")
        return f"An error occurred while generating the response: {e}"


# --- UI Layout ---
st.title("🩺 Medical PDF Analysis Dashboard")
add_vertical_space(1)

tab_upload, tab_ask, tab_history = st.tabs(["Upload Documents", "❓ Ask a Question", "📜 History"])

# --- Upload Tab ---
with tab_upload:
    st.header("Upload Medical PDFs")
    if not PDF_PREPROCESSING_AVAILABLE:
        st.error("The `pdf_preprocessing.py` script was not found. Please place it in the same directory to enable uploads.")
    else:
        uploaded_pdfs = st.file_uploader(
            "Choose PDF files to analyze",
            type="pdf",
            accept_multiple_files=True
        )

        if st.button("Process Uploaded Files") and uploaded_pdfs:
            with st.spinner("Processing PDFs... This may take a moment."):
                for pdf in uploaded_pdfs:
                    file_name = pdf.name
                    if file_name in st.session_state.uploaded_files:
                        st.warning(f"'{file_name}' has already been processed. Skipping.")
                        continue
                    
                    try:
                        # Save the file temporarily to be read by PyPDF2/pdfplumber
                        with open(file_name, "wb") as f:
                            f.write(pdf.getbuffer())
                        
                        st.info(f"Extracting text from '{file_name}'...")
                        text = extract_text_robust(file_name)

                        if text:
                            st.info(f"Chunking text for '{file_name}'...")
                            chunks = chunk_text(text)
                            
                            st.info(f"Storing chunks in database for '{file_name}'...")
                            store_in_chroma(chunks, file_name)
                            
                            st.session_state.uploaded_files.append(file_name)
                            st.success(f"✅ Successfully processed and indexed '{file_name}'!")
                        else:
                            st.error(f"Failed to extract any text from '{file_name}'.")

                        os.remove(file_name) # Clean up the temp file
                    
                    except Exception as e:
                        st.error(f"An error occurred while processing '{file_name}': {e}")
                        if os.path.exists(file_name):
                            os.remove(file_name)


# --- Ask a Question Tab ---
with tab_ask:
    st.header("Query Your Documents")
    if not chroma_client:
        st.error("Cannot connect to document database (ChromaDB). Please check configuration.")
    else:
        try:
            collections = chroma_client.list_collections()
            collection_names = [c.name for c in collections]
        except Exception:
            collections = []
            collection_names = []

        if not collection_names:
            st.warning("No document collections found. Please upload PDFs in the 'Upload' tab first.")
        else:
            selected_collection = st.selectbox(
                "Choose a document collection to query:",
                options=collection_names,
                index=0
            )
            
            question = st.text_input("Enter your question about the selected document:", key="question_input")

            if st.button("Get Answer"):
                if not question:
                    st.warning("Please enter a question.")
                elif not llm:
                    st.error("LLM is not available. Cannot proceed.")
                else:
                    with st.spinner("Searching for relevant information and generating answer..."):
                        # 1. Retrieve chunks from ChromaDB
                        retrieved_chunks, error = query_chroma(question, selected_collection)
                        
                        if error:
                            st.error(error)
                        elif not retrieved_chunks or not retrieved_chunks['documents'][0]:
                            st.warning("Could not find any relevant information in the document to answer this question.")
                        else:
                            # 2. Generate response with LLM
                            context_docs = retrieved_chunks['documents'][0]
                            answer = generate_response(llm, question, context_docs)
                            
                            # 3. Display the answer and sources
                            st.subheader("Answer")
                            st.markdown(f'<div class="response-box"><p>{answer}</p></div>', unsafe_allow_html=True)
                            add_vertical_space(1)
                            
                            # 4. Display expandable source chunks
                            st.subheader("Retrieved Sources")
                            st.info("The answer was generated based on the following text chunks from the document:")
                            
                            for i, (doc, meta) in enumerate(zip(retrieved_chunks['documents'][0], retrieved_chunks['metadatas'][0])):
                                with st.expander(f"Source Chunk {i+1} (From: {meta.get('source', 'N/A')})"):
                                    st.write(doc)

                            # 5. Update history
                            st.session_state.history.append({
                                "collection": selected_collection,
                                "question": question,
                                "answer": answer
                            })

# --- History Tab ---
with tab_history:
    st.header("Question & Answer History")
    if not st.session_state.history:
        st.info("No questions have been asked yet.")
    else:
        for i, entry in enumerate(reversed(st.session_state.history)):
            st.markdown(f"**Collection:** `{entry['collection']}`")
            st.markdown(f"**Q{len(st.session_state.history)-i}:** {entry['question']}")
            st.markdown(f"**A:** {entry['answer']}")
            st.divider()
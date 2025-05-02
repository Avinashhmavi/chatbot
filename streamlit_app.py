import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PDFPlumberLoader, UnstructuredExcelLoader, CSVLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage
import os
import tempfile
import shutil
import warnings

warnings.simplefilter("ignore", category=UserWarning)

# Set page configuration
st.set_page_config(page_title="Document Q&A", layout="wide")

# Initialize session state for vector store, memory, and RAG chain
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".xlsx", ".xls", ".csv", ".doc", ".docx"}

def process_file(file_path: str, file_extension: str):
    """Process uploaded file based on its extension."""
    documents = []
    
    if file_extension == ".pdf":
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()
    elif file_extension in [".xlsx", ".xls"]:
        loader = UnstructuredExcelLoader(file_path)
        documents = loader.load()
    elif file_extension == ".csv":
        loader = CSVLoader(file_path)
        documents = loader.load()
    elif file_extension in [".doc", ".docx"]:
        loader = UnstructuredWordDocumentLoader(file_path)
        documents = loader.load()
    
    return documents

def initialize_rag_chain(documents):
    """Initialize RAG chain with processed documents."""
    # Access API key from Streamlit secrets
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    
    # Split text into chunks for embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    # Create embeddings and store them in FAISS
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    st.session_state.vector_store = FAISS.from_documents(docs, embeddings)
    retriever = st.session_state.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10})
    
    # Create the ConversationalRetrievalChain
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai_api_key)
    st.session_state.rag_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=st.session_state.memory
    )

# Initialize a separate LLM for general questions
@st.cache_resource
def get_general_llm():
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai_api_key)

# Streamlit UI
st.title("💬🤖 ChatBOT")

# File upload section
st.header("Upload Document")
uploaded_file = st.file_uploader(
    "Upload a document (PDF, Excel, CSV, Word)", 
    type=["pdf", "xlsx", "xls", "csv", "doc", "docx"]
)

if uploaded_file:
    try:
        # Check file extension
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension not in SUPPORTED_EXTENSIONS:
            st.error("Unsupported file format. Supported formats: PDF, Excel, CSV, Word")
        else:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                shutil.copyfileobj(uploaded_file, temp_file)
                temp_file_path = temp_file.name
            
            try:
                # Process the uploaded file
                documents = process_file(temp_file_path, file_extension)
                
                if not documents:
                    st.error("No content could be extracted from the file")
                else:
                    # Initialize RAG chain with new documents
                    initialize_rag_chain(documents)
                    st.success(f"File {uploaded_file.name} uploaded and processed successfully")
            
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Chat interface
st.header("Ask a Question")
question = st.text_input("Enter your question:")
if st.button("Submit"):
    if question:
        try:
            if st.session_state.rag_chain is None:
                # Use general LLM for questions without uploaded documents
                general_llm = get_general_llm()
                response = general_llm.invoke([HumanMessage(content=question)])
                answer = response.content
            else:
                # Use RAG chain for document-related questions
                response = st.session_state.rag_chain.invoke({"question": question})
                answer = response["answer"]
            
            # Update chat history
            st.session_state.chat_history.append(("You", question))
            st.session_state.chat_history.append(("Assistant", answer))
        
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")

# Display chat history
st.header("Chat History")
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**You**: {message}")
    else:
        st.markdown(f"**Assistant**: {message}")

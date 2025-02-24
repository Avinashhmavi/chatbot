import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
import os
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
import tempfile
from groq import Groq  # Hypothetical Groq API client
from typing import List, Dict

# Initialize Groq client (you'd need API key from xAI)
client = Groq(api_key="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")

# Function to process different file types
def process_uploaded_file(file) -> str:
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name

    try:
        # Handle different file types
        if file.name.endswith('.pdf'):
            reader = PdfReader(tmp_file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        
        elif file.name.endswith('.csv'):
            df = pd.read_csv(tmp_file_path)
            text = df.to_string()
        
        elif file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(tmp_file_path)
            text = df.to_string()
        
        elif file.name.endswith('.docx'):
            doc = Document(tmp_file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        
        else:
            # For other text-based files
            loader = UnstructuredFileLoader(tmp_file_path)
            docs = loader.load()
            text = "\n".join([doc.page_content for doc in docs])
        
        return text
    
    finally:
        os.unlink(tmp_file_path)

# Function to get response from Groq
def get_grok_response(query: str, context: str = "") -> str:
    try:
        if context:
            prompt = f"Context: {context}\n\nQuestion: {query}"
        else:
            prompt = query
            
        response = client.chat.completions.create(
            model="grok-3",  # Hypothetical model name
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that can answer questions based on provided context or general knowledge."},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit app
def main():
    st.title("Grok Chatbot with Document Upload")
    
    # Initialize session state for chat history and uploaded content
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'uploaded_content' not in st.session_state:
        st.session_state.uploaded_content = ""

    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload files (PDF, CSV, Excel, Word, etc.)",
            accept_multiple_files=True
        )
        
        if uploaded_files:
            with st.spinner("Processing files..."):
                for file in uploaded_files:
                    content = process_uploaded_file(file)
                    st.session_state.uploaded_content += f"\n\nFile: {file.name}\n{content}"
                st.success("Files processed successfully!")

    # Chat interface
    st.header("Chat with Grok")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask anything..."):
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Add to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Get and display Grok's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_grok_response(
                    prompt,
                    st.session_state.uploaded_content
                )
                st.write(response)
        
        # Add response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

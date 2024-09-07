import os
from dotenv import load_dotenv
import streamlit as st
import requests
import fitz  # PyMuPDF
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.groq import Groq

# Load environment variables
load_dotenv()

# Configuration
GITHUB_PDF_URL = "https://github.com/SauravSrivastav/sauravsrivastavbot/raw/main/Data/SauravSrivastav_cv.pdf"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@st.cache_resource
def load_and_process_pdf():
    # Fetch PDF from GitHub
    response = requests.get(GITHUB_PDF_URL)
    response.raise_for_status()
    
    # Extract text from PDF
    with fitz.open(stream=io.BytesIO(response.content), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    
    return vectorstore

def get_qa_chain(vectorstore):
    llm = Groq(api_key=GROQ_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

def main():
    st.title("Saurav's CV Q&A with Groq")

    # Load and process PDF
    vectorstore = load_and_process_pdf()
    qa_chain = get_qa_chain(vectorstore)

    # Chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi there! I'm here to answer questions about Saurav Srivastav based on his CV. What would you like to know?"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask your question here:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            result = qa_chain({"query": prompt})
            response = result['result']
            message_placeholder.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

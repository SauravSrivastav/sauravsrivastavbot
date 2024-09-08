import os
import io
import streamlit as st
import requests
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Configuration
GITHUB_PDF_URL = "https://github.com/SauravSrivastav/sauravsrivastavbot/raw/main/Data/SauravSrivastav_cv.pdf"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@st.cache_resource
def load_and_process_pdf():
    """
    Download and process the PDF, then create and return a FAISS vector store.
    """
    try:
        response = requests.get(GITHUB_PDF_URL)
        response.raise_for_status()

        pdf_reader = PdfReader(io.BytesIO(response.content))
        text = " ".join(page.extract_text() for page in pdf_reader.pages)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        vectorstore = FAISS.from_texts(chunks, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def get_qa_chain(vectorstore):
    """
    Create and return a question-answering chain using the Groq API and FAISS vector store.
    """
    try:
        llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")

        template = """You are Saurav Srivastav, a professional with experience in Cloud and DevOps engineering.
        Use the following pieces of context to answer the question at the end in a natural and conversational manner.
        If you don't know the answer or can't find relevant information in the context, just say that you don't have enough information to answer the question accurately.

        Context:
        {context}

        Question: {question}
        Answer: """

        PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        return None

def main():
    st.title("Chat with Saurav Srivastav")

    vectorstore = load_and_process_pdf()
    if vectorstore is None:
        st.error("Failed to load and process the CV. Please try again later.")
        return

    qa_chain = get_qa_chain(vectorstore)
    if qa_chain is None:
        st.error("Failed to initialize the Q&A system. Please check your API key and try again.")
        return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask me anything about my experiences and achievements:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                with st.spinner("Searching for an answer..."):
                    response = qa_chain({"query": prompt})
                    if response['result']:
                        full_response = response['result']
                    else:
                        full_response = "I'm sorry, I don't have enough information to answer that question accurately based on my experiences and achievements."

                message_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"An error occurred while processing your request: {str(e)}"
                message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    st.markdown("---")
    st.markdown("**Note:** This chatbot provides information based on Saurav Srivastav's experiences and achievements.")

if __name__ == "__main__":
    main()

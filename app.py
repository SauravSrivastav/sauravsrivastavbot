import os
import io
import streamlit as st
import requests
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import logging
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
GITHUB_PDF_URL = "https://github.com/SauravSrivastav/sauravsrivastavbot/raw/main/Data/SauravSrivastav_cv.pdf"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLAMA_MODEL = "llama-3.1-70b-versatile"

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not set in the environment variables. Please set it and restart the application.")
    st.stop()

@st.cache_resource
def load_and_process_pdf():
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
        return vectorstore, text
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return None, None

def alfred_response(user_prompt: str, chat_history, resume_text: str, vectorstore):
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        
        # Retrieve relevant context from the vectorstore
        relevant_docs = vectorstore.similarity_search(user_prompt, k=3)
        relevant_context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Construct system message with resume context, relevant information, and chat history
        system_message = f"""You are Alfred, Saurav Srivastav's AI assistant. Use this resume context: {resume_text}

Relevant information from the resume:
{relevant_context}

Guidelines:
1. Always stay in character as Alfred, Saurav's AI assistant.
2. Provide information primarily about Saurav's professional background, skills, and achievements.
3. If asked about topics unrelated to Saurav, provide a brief, informative response and then try to steer the conversation back to Saurav's expertise.
4. Do not share personal contact information.
5. Maintain context throughout the conversation and refer back to previous points when relevant.
6. If you don't have specific information about an aspect of Saurav's background, say so honestly.
7. Use the chat history to maintain consistency in your responses.

Chat History:
"""
        for role, content in chat_history:
            system_message += f"{role.capitalize()}: {content}\n"
        
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            model=LLAMA_MODEL,
            temperature=0.1,
            max_tokens=1024,
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error in alfred_response: {str(e)}")
        return "I apologize, but I'm having trouble processing your request at the moment. How about we discuss Saurav's expertise in Cloud engineering? Would you like to know more about his experience with Kubernetes or containerization?"

def main():
    st.set_page_config(page_title="Chat with Alfred - Saurav's AI Assistant", page_icon="🤖", layout="centered")

    st.title("Chat with Alfred, Saurav's AI Assistant")

    # Custom CSS for better UI
    st.markdown("""
        <style>
        .stTextInput > div > div > input {
            caret-color: #FF4B4B;
        }
        .stButton > button {
            background-color: #FF4B4B;
            color: white;
        }
        .stButton > button:hover {
            background-color: #FF6B6B;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.title("🦙💬 Alfred")
        st.caption("🚀 A Chatbot for Saurav's Professional Insights and More")
        st.markdown("📝 [Access resume here](https://github.com/SauravSrivastav/sauravsrivastavbot/raw/main/Data/SauravSrivastav_cv.pdf)")
        st.markdown("👾 [Access GitHub here](https://github.com/SauravSrivastav)")

        if st.button('Clear Chat History'):
            st.session_state.messages = []
            st.experimental_rerun()

    vectorstore, resume_text = load_and_process_pdf()
    if vectorstore is None or resume_text is None:
        st.error("Failed to load and process the CV. Please try again later.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! I'm Alfred, Saurav Srivastav's personal AI assistant. I'm here to provide information about Saurav's professional background in Cloud and DevOps engineering. How can I assist you today?"
        }]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about Saurav's professional background or any other topic"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                with st.spinner("Alfred is thinking..."):
                    chat_history = [(m["role"], m["content"]) for m in st.session_state.messages]
                    full_response = alfred_response(prompt, chat_history, resume_text, vectorstore)

                message_placeholder.markdown(full_response)
            except Exception as e:
                logger.error(f"Error in main chat loop: {str(e)}")
                full_response = "I apologize, but I'm having trouble processing your request at the moment. How about we discuss Saurav's expertise in Cloud engineering? Would you like to know more about his experience with Kubernetes or containerization?"
                message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

    st.markdown("---")
    st.caption("Alfred is here to assist you with information about Saurav Srivastav's professional background in Cloud and DevOps engineering.")

if __name__ == "__main__":
    main()

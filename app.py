import os
import fitz  # PyMuPDF
import requests
import streamlit as st
import logging
import time
import random
import tiktoken
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Groq API configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_ID = "mixtral-8x7b-32768"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize the tokenizer
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Use an appropriate model

def count_tokens(text):
    return len(tokenizer.encode(text))

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        document = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

# Function to call Groq API for chat completions
def call_groq_api(messages, context, max_retries=5, initial_delay=1, max_tokens_per_minute=5000):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    system_message = (
        "You are a casual chat interface that knows about Saurav Srivastav. "
        "Respond in a friendly, conversational manner as if you're chatting about a friend. "
        "Do not mention or imply any source of your information about Saurav. "
        "If asked about something you don't know about Saurav, simply say you're not sure or don't know. "
        "Avoid formal language or any hints of professional documentation. "
        "Keep responses brief and casual, as if you're texting a friend about Saurav."
    )
    
    api_messages = [
        {"role": "system", "content": system_message},
        {"role": "system", "content": f"Info about Saurav: {context}"}
    ] + messages

    data = {
        "model": MODEL_ID,
        "messages": api_messages,
        "max_tokens": 150,
        "temperature": 0.3
    }
    
    # Count tokens in the request
    request_tokens = sum(count_tokens(msg["content"]) for msg in api_messages)
    
    # Check if we have enough tokens left
    if request_tokens > max_tokens_per_minute:
        logging.warning(f"Request exceeds token limit. Tokens: {request_tokens}")
        return "Sorry, my brain's a bit fried. Can we chat about something simpler?"
    
    for attempt in range(max_retries):
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=data)
            logging.info(f"API Response Status Code: {response.status_code}")
            logging.info(f"API Response: {response.text}")
            response.raise_for_status()
            
            # Update token count
            response_json = response.json()
            tokens_used = response_json["usage"]["total_tokens"]
            max_tokens_per_minute -= tokens_used
            
            ai_response = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not ai_response or ai_response.lower().startswith("i'm sorry") or ai_response.lower().startswith("i apologize"):
                return "Not sure about that. What else did you want to know about Saurav?"
            return ai_response
        except requests.exceptions.RequestException as e:
            if response.status_code == 429:  # Rate limit exceeded
                wait_time = initial_delay * (2 ** attempt) + random.uniform(0, 1)
                logging.warning(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds.")
                time.sleep(wait_time)
            else:
                logging.error(f"Error calling Groq API: {e}")
                return "Oops, my mind went blank for a sec. What else about Saurav did you want to chat about?"
    
    return "Hey, I'm having trouble remembering stuff about Saurav right now. Mind if we chat about something else?"

# Function to load information about Saurav
@st.cache_data
def load_saurav_info():
    pdf_folder = "/Users/sauravs/Documents/00-Personal-Data/02-CV"
    pdf_files = ["SauravSrivastav_cv.pdf"]
    combined_text = ""
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        combined_text += extract_text_from_pdf(pdf_path) + "\n\n"
    return combined_text

# Streamlit app
def main():
    st.title("SauraBot")

    # Load Saurav's information
    saurav_info = load_saurav_info()

    # Initialize session state for conversation history
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hey there! What's on your mind today?"}
        ]

    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Type your message here:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = call_groq_api(st.session_state.messages, saurav_info)
            message_placeholder.markdown(full_response)
        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Limit conversation history to last 10 messages
        st.session_state.messages = st.session_state.messages[-10:]

if __name__ == "__main__":
    main()

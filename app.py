import os
import requests
import streamlit as st
import logging
import time
import random
from dotenv import load_dotenv
import io
import fitz  # PyMuPDF

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Groq API configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_ID = "mixtral-8x7b-32768"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# GitHub PDF URL
GITHUB_PDF_URL = "https://github.com/SauravSrivastav/sauravsrivastavbot/raw/main/Data/SauravSrivastav_cv.pdf"

@st.cache_resource
def load_cv_content():
    try:
        # Fetch PDF from GitHub
        response = requests.get(GITHUB_PDF_URL)
        response.raise_for_status()
        
        # Read PDF content
        with fitz.open(stream=io.BytesIO(response.content), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        
        return text
    except Exception as e:
        logging.error(f"Error loading CV content: {e}")
        return f"Failed to load CV content. Error: {str(e)}"

def call_groq_api(messages, context, max_retries=5, initial_delay=1):
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
        "max_tokens": 1500,
        "temperature": 0.1
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=data)
            logging.info(f"API Response Status Code: {response.status_code}")
            logging.info(f"API Response: {response.text}")
            response.raise_for_status()
            
            response_json = response.json()
            ai_response = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not ai_response or ai_response.lower().startswith("i'm sorry") or ai_response.lower().startswith("i apologize"):
                return "Not sure about that. What else did you want to know about Saurav?"
            return ai_response
        except requests.RequestException as e:
            if response.status_code == 429:  # Rate limit exceeded
                wait_time = initial_delay * (2 ** attempt) + random.uniform(0, 1)
                logging.warning(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds.")
                time.sleep(wait_time)
            else:
                logging.error(f"Error calling Groq API: {e}")
                return "Oops, my mind went blank for a sec. What else about Saurav did you want to chat about?"
    
    return "Hey, I'm having trouble remembering stuff about Saurav right now. Mind if we chat about something else?"

def main():
    st.title("SauraBot")

    # Load Saurav's information
    saurav_info = load_cv_content()

    if saurav_info.startswith("Failed to load CV content"):
        st.error(saurav_info)
        st.stop()

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
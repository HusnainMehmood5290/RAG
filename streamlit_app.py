import streamlit as st
import time

# Set up the page configuration
st.set_page_config(page_title="DEMO Chatbot", layout="centered")

st.title("Demo Chatbot")

# Initialize session state for messages if it doesn't exist yet
if "messages" not in st.session_state:
    st.session_state.messages = []


# Get user input using chat input widget
user_input = st.chat_input("Say something...")

if user_input:
    # Add the user's message to session state and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate a simple bot response (here, just echoing the input)
    bot_response = f"Echo: {user_input}"
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)

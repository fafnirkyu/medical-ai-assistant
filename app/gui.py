import streamlit as st
import requests

st.set_page_config(page_title="Medical AI Assistant", page_icon="🩺")

st.title("🩺 Medical RAG Assistant")
st.markdown("Ask a medical question. The AI will search the database and provide an answer.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How can I help you today?"):
    # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Searching medical records and thinking..."):
            try:
                # Call our FastAPI backend
                response = requests.get(f"http://127.0.0.1:8000/ask?query={prompt}")
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    context = data["source_context"]
                    
                    st.markdown(answer)
                    with st.expander("View Source Context"):
                        st.info(context)
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error("Backend error. Is the API running?")
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")
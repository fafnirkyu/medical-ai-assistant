import streamlit as st
import requests

st.set_page_config(page_title="Medical AI Assistant", page_icon="🩺")

st.title("Medical RAG Assistant by Antonio Borges")
st.markdown("THIS IS A PROOF OF CONCEPT, REMEMBER TO CONSULT A REAL PHYSICIAN.")
st.markdown("Ask a medical question. The AI will search the database and provide an answer.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching medical records and thinking..."):
            try:
                response = requests.get(f"http://backend:8000/ask?query={prompt}")
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    context = data.get("source", "No source provided")
                    score = data.get("confidence", 0.0)

                    st.markdown(answer)
                    
                    if score > 0.80:
                        st.success(f"High Confidence: {int(score*100)}%")
                    elif score > 0.50:
                        st.warning(f"Moderate Confidence: {int(score*100)}%")
                    else:
                        st.error(f"Low Confidence: {int(score*100)}% - Use caution.")
                    
                    st.progress(score, text=f"Confidence Score: {int(score*100)}%") 
                    with st.expander("View Source Context"):
                        st.info(context)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"{answer}\n\n*Source: {context}*"
                    })
                else:
                    st.error("Backend error. Is the API running?")
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")

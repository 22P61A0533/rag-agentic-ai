import streamlit as st
from src.graph import ask_question

st.set_page_config(page_title="Agentic AI RAG Chatbot")

st.title("Agentic AI RAG Chatbot")
st.write("Ask questions based only on the provided Agentic AI eBook.")

question = st.text_input("Enter your question:")

if st.button("Ask") and question:
    with st.spinner("Generating answer..."):
        result = ask_question(question)

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Confidence Score")
    st.write(result["confidence_score"])

    st.subheader("Retrieved Context")
    for i, chunk in enumerate(result["context"], 1):
        with st.expander(f"Context Chunk {i}"):
            st.write(chunk)

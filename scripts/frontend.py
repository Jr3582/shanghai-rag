import streamlit as st
import requests

st.title("Battle of Shanghai RAG Assistant")

query = st.text_input("Ask a question about the Battle of Shanghai:")

if st.button("Ask"):
    response = requests.post("https://shanghai-rag.onrender.com/ask", json={"query": query})
    answer = response.json()["answer"]
    st.write(answer)
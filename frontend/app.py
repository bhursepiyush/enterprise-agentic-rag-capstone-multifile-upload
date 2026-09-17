import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Enterprise QA Agent", layout="wide")
st.title("🤖 Autonomous Enterprise Document QA Agent")

with st.sidebar:
    st.header("Upload Enterprise Files")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, CSV, Excel", 
        type=["pdf", "txt", "csv", "xlsx"], 
        accept_multiple_files=True
    )
    if st.button("Index Documents") and uploaded_files:
        for file in uploaded_files:
            res = requests.post(f"{BACKEND_URL}/upload", files={"file": (file.name, file.getvalue())})
            if res.status_code == 200:
                st.success(f"{file.name}: {res.json().get('chunks_stored')} chunks indexed.")
            else:
                st.error(f"Failed to process {file.name}")

st.subheader("Ask Document Questions")
user_query = st.text_input("Enter your question:")
if st.button("Submit Query") and user_query:
    with st.spinner("Agent planning and retrieving context..."):
        res = requests.post(f"{BACKEND_URL}/query", json={"question": user_query})
        if res.status_code == 200:
            data = res.json()
            st.markdown("### Answer")
            st.write(data.get("answer"))

            with st.expander("Show Agent Reasoning & Context"):
                st.json(data.get("plan"))
                st.write("**Retrieved Chunks:**")
                for i, chunk in enumerate(data.get("retrieved_context", [])):
                    st.info(f"Chunk {i+1}: {chunk}")
        else:
            st.error("Error communicating with backend.")
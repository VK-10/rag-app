import os
import streamlit as st
from docs_chat_utility import get_answer
import tempfile

working_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Chat with Doc",
    page_icon="📊",
    layout="centered",
)

st.title("Doc Chat Utility")

uploaded_file = st.file_uploader("Upload your file", type=["txt", "docx", "pdf"])
user_query = st.text_input("Enter your query")

if st.button("Generate"):
    # Ensure that a file was uploaded
    if uploaded_file is not None:
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as temp_file:
            # Write the file data to the temporary file
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name  # Get the temporary file path

        # Pass the temporary file path to get_answer
        ans = get_answer(temp_file_path, user_query)

        # Remove the temporary file after processing
        os.remove(temp_file_path)

        st.success(ans)
    else:
        st.error("Please upload a file before clicking Generate.")

import os
from dotenv import load_dotenv
import streamlit as st
import openai
import pandas as pd
from collections import defaultdict
from functools import partial
from io import StringIO
from PyPDF2 import PdfReader


st.set_page_config(page_title="LLM Document Analyzer", layout="wide")

def pdf_to_text(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text


def process_document(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()



def analyze_documents(files):
    summaries = []
    for file in files:
        if file.type == "application/pdf":
            content = pdf_to_text(file)
        else:
            try:
                content = file.getvalue().decode("utf-8")
            except UnicodeDecodeError:
                st.error(f"Error: Unable to decode the file {file.name}. Please make sure it is in UTF-8 format.")
                return None, None, None

        summary_prompt = f"Summarize the following document in 5 dot points: {content}"
        summary = process_document(summary_prompt)
        summaries.append(summary)

    similarities_prompt = f"Find similarities between these documents in 5 dot points: {summaries}"
    similarities = process_document(similarities_prompt)

    differences_prompt = f"Find differences between these documents: {summaries}"
    differences = process_document(differences_prompt)

    return summaries, similarities, differences

def main():
    st.title("LLM Document Analyzer")

    st.sidebar.header("OpenAI API Key")
    api_key = st.sidebar.text_input("Enter your API key", type="password")

    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        return

    openai.api_key = api_key

    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=["txt", "pdf"])

    if uploaded_files:
        if st.button("Analyze Documents"):
            summaries, similarities, differences = analyze_documents(uploaded_files)

            st.header("Document Summaries")
            for i, summary in enumerate(summaries):
                st.subheader(f"Document {i + 1}")
                st.write(summary)

            st.header("Similarities")
            st.write(similarities)

            st.header("Differences")
            st.write(differences)
        else:
            st.write("Click the 'Analyze Documents' button to process the uploaded files.")
    else:
        st.write("Upload one or more files to begin analyzing.")

if __name__ == "__main__":
    main()

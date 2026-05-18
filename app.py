

import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import os

# --- Page Config ---
st.set_page_config(page_title="AskMyPDF RAG", page_icon="📄")

# --- Gemini API Key ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except:
    st.error("Bhai GOOGLE_API_KEY nahi mila. Streamlit → App settings → Secrets me daal de: GOOGLE_API_KEY = 'tera-key'")
    st.stop()

# --- Functions ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_embeddings(text_chunks):
    model = 'text-embedding-004'
    embeddings = genai.embed_content(model=model, content=text_chunks, task_type="retrieval_document")["embedding"]
    return np.array(embeddings)

def get_vector_store(text_chunks):
    embeddings = get_embeddings(text_chunks)
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, text_chunks

def get_answer(question, index, text_chunks):
    question_embedding = genai.embed_content(model='text-embedding-004', content=question, task_type="retrieval_query")["embedding"]
    D, I = index.search(np.array([question_embedding]), k=4)
    relevant_chunks = [text_chunks[i] for i in I[0]]
    context = "\n\n".join(relevant_chunks)

    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Context:\n {context}\n
    Question: {question}\n
    Answer the question based on the context above. If answer is not in context, say "PDF me ye nahi mila".
    Answer:
    """
    response = model.generate_content(prompt)
    return response.text

# --- UI ---
st.title("📄 AskMyPDF - RAG with Gemini")
st.write("PDF upload kar aur usse sawaal pooch")

with st.sidebar:
    st.header("Upload PDF")
    pdf_docs = st.file_uploader("PDF file daal", accept_multiple_files=True, type="pdf")
    if st.button("Process Karo"):
        if pdf_docs:
            with st.spinner("PDF padh raha hun..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                st.session_state.index, st.session_state.chunks = get_vector_store(text_chunks)
                st.success("Ho gaya! Ab sawaal pooch")
        else:
            st.warning("Pehle PDF to daal bhai")

user_question = st.text_input("PDF se kya poochna hai?")

if user_question:
    if "index" not in st.session_state:
        st.warning("Pehle PDF process kar bhai")
    else:
        with st.spinner("Jawab dhoond raha hun..."):
            answer = get_answer(user_question, st.session_state.index, st.session_state.chunks)
            st.write("Jawab:", answer)

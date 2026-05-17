import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import os

# Page config
st.set_page_config(page_title="AskMyPDF - RAG App", page_icon="📄", layout="wide")

st.title("📄 AskMyPDF - Chat with your PDF")
st.markdown("Upload a PDF and ask questions. Powered by Gemini + FAISS")

# Sidebar for API Key
with st.sidebar:
    st.header("⚙️ Settings")
    gemini_api_key = st.text_input("Enter Gemini API Key", type="password",
                                   help="Get your key from https://aistudio.google.com/app/apikey")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")

# Function to extract text from PDF
def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create embeddings using Gemini
def get_embeddings(texts):
    embeddings = []
    for text in texts:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        embeddings.append(response['embedding'])
    return np.array(embeddings)

# Function to create FAISS vector store
def create_vector_store(text_chunks):
    embeddings = get_embeddings(text_chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

# Function to get relevant chunks
def get_relevant_chunks(query, index, text_chunks, k=3):
    query_embedding = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )['embedding']

    query_embedding = np.array([query_embedding])
    distances, indices = index.search(query_embedding, k)

    relevant_chunks = [text_chunks[i] for i in indices[0]]
    return relevant_chunks

# Function to generate answer
def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Answer the question based only on the context provided below.
If the answer is not in the context, say "I couldn't find the answer in the provided PDF."

Context:
{context}

Question: {query}

Answer:"""

    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# Main app logic
if not gemini_api_key:
    st.warning("⚠️ Please enter your Gemini API Key in the sidebar to continue")
    st.stop()

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner("Reading PDF..."):
        raw_text = get_pdf_text(uploaded_file)

    if raw_text:
        st.success(f"PDF uploaded! Total characters: {len(raw_text)}")

        with st.spinner("Creating embeddings... This may take a minute"):
            text_chunks = get_text_chunks(raw_text)
            vector_index, _ = create_vector_store(text_chunks)
            st.session_state['vector_index'] = vector_index
            st.session_state['text_chunks'] = text_chunks

        st.success("✅ PDF processed! You can now ask questions")

        query = st.text_input("Ask a question about your PDF:")

        if query:
            with st.spinner("Thinking..."):
                relevant_chunks = get_relevant_chunks(
                    query,
                    st.session_state['vector_index'],
                    st.session_state['text_chunks']
                )
                answer = generate_answer(query, relevant_chunks)

            st.markdown("### 🤖 Answer:")
            st.write(answer)

            with st.expander("📚 See relevant PDF chunks used"):
                for i, chunk in enumerate(relevant_chunks):
                    st.write(f"**Chunk {i+1}:** {chunk[:300]}...")
    else:
        st.error("Could not extract text from PDF. Is it a scanned PDF?")
else:
    st.info("👆 Upload a PDF file to get started")
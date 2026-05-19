import streamlit as st 
from google import genai
from PyPDF2 import PdfReader

st.set_page_config(page_title="AskMyPDF", page_icon="📄", layout="wide")
st.title("AskMyPDF - PDF se Sawal Puchiye 📄")

try:
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception as e:
    st.error("Secrets me GOOGLE_API_KEY nahi mili. Streamlit Cloud > App settings > Secrets me daal de.")
    st.stop()

st.subheader("Step 1: PDF Upload Karein")
pdf_file = st.file_uploader("PDF file choose karein", type="pdf")

if pdf_file is not None:
    with st.spinner("PDF padh raha hu..."):
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    
    if text:
        st.success(f"PDF process ho gaya! Total {len(pdf_reader.pages)} pages mile.")
        st.subheader("Step 2: Sawal Puchiye")
        user_question = st.text_input("Apna sawal yaha likhein:")
        
        if st.button("Jawab Do") and user_question:
            with st.spinner("Jawab dhoond raha hu..."):
                try:
                    prompt = f"""
                    Niche diye gaye PDF context ke basis pe user ke sawal ka jawab do.
                    Agar jawab context me nahi hai to bol do "Ye PDF me nahi hai".
                    
                    PDF Context:
                    {text[:30000]}
                    
                    Sawal: {user_question}
                    Jawab:
                    """
                    
                    response = client.models.generate_content(
                        model="gemini-2.5-flash", 
                        contents=prompt
                    )
                    st.write("### Jawab:")
                    st.write(response.text)
                    
                except Exception as e:
                    st.error(f"Gemini se error aaya: {e}")
    else:
        st.error("Is PDF se text nahi nikal paya. Koi aur PDF try karo.")
else:
    st.info("Shuru karne ke liye upar PDF upload karo.")

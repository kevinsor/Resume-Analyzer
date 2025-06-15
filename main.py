import streamlit as st

# Set page config FIRST
st.set_page_config(page_title="Resume Analyzer (Transformer-Powered)", layout="centered")

import pdfplumber
import docx
from sentence_transformers import SentenceTransformer, util


# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


model = load_model()


# Read PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


# Read docx
def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])


st.title("📄 AI Resume Analyzer")
st.write("Upload your resume and compare it with a job description using transformer-based semantic similarity.")

uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])

job_desc = st.text_area("Paste the Job Description Here", height=200)

if uploaded_file and job_desc.strip():
    with st.spinner("Analyzing..."):
        # Extract resume text
        if uploaded_file.name.endswith(".pdf"):
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_text = extract_text_from_docx(uploaded_file)

        # Display raw text (optional)
        with st.expander("📄 Show Extracted Resume Text"):
            st.write(resume_text)

        # Generate sentence embeddings
        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        job_embedding = model.encode(job_desc, convert_to_tensor=True)

        # Compute cosine similarity
        similarity_score = util.pytorch_cos_sim(resume_embedding, job_embedding).item()

        # Display Score
        st.subheader("📊 Semantic Similarity Score")
        st.metric(label="Resume vs Job Description", value=f"{similarity_score * 100:.2f}%")

        # Interpret the score
        if similarity_score > 0.85:
            st.success("🔥 Excellent match! Your resume strongly aligns with the job description.")
        elif similarity_score > 0.70:
            st.info("✅ Decent match. Some tailoring could improve alignment.")
        else:
            st.warning("⚠️ Low similarity. Consider revising your resume to better fit the job.")

        # Show both texts side-by-side for comparison
        with st.expander("📝 Job Description and Resume Side-by-Side"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Job Description**")
                st.write(job_desc)
            with col2:
                st.markdown("**Resume Text**")
                st.write(resume_text)

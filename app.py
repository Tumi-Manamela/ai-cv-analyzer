import streamlit as st
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI CV Analyzer",
    page_icon="📄",
    layout="centered"
)

st.title("📄 AI CV Analyzer")
st.write("Upload your CV and compare it against a job description.")

def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)

    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "

    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9+#. ]", " ", text)
    return text

skills_list = [
    "python", "java", "sql", "excel", "machine learning",
    "data analysis", "data visualization", "streamlit",
    "pandas", "numpy", "scikit-learn", "tensorflow",
    "power bi", "tableau", "communication", "problem solving",
    "problem-solving", "teamwork", "leadership", "html", "css",
    "javascript", "git", "github", "database", "database systems",
    "cloud computing", "cybersecurity", "nlp",
    "artificial intelligence", "programming", "statistics",
    "data manipulation", "web technology", "networks",
    "big data", "iot", "computer literacy", "research skills",
    "time management", "adaptability", "accountability"
]

uploaded_cv = st.file_uploader(
    "Upload your CV as a PDF",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste the Job Description here"
)

if st.button("Analyze CV"):

    if uploaded_cv and job_description:

        cv_text = extract_text_from_pdf(uploaded_cv)
        

        cleaned_cv = clean_text(cv_text)
        cleaned_job = clean_text(job_description)

        documents = [cleaned_cv, cleaned_job]

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        tfidf_matrix = vectorizer.fit_transform(documents)

        similarity_score = cosine_similarity(
            tfidf_matrix[0:1],
            tfidf_matrix[1:2]
        )[0][0]

        cv_skills = []
        job_skills = []

        for skill in skills_list:

            if skill in cleaned_cv:
                cv_skills.append(skill)

            if skill in cleaned_job:
                job_skills.append(skill)

        missing_skills = sorted(
    list(set(job_skills) - set(cv_skills))
)

    st.subheader("Results")

    score_percentage = round(similarity_score * 100, 2)

    st.metric("CV Match Score", f"{score_percentage}%")
    st.progress(similarity_score)

    if score_percentage >= 70:
            st.success("Strong match")
    elif score_percentage >= 40:
            st.warning("Moderate match")
    else:
            st.error("Low match")

    matched_skills = sorted(
            list(set(cv_skills) & set(job_skills))
        )

    col1, col2 = st.columns(2)

    with col1:
            st.subheader("Matched Skills")
            st.write(f"{len(matched_skills)} matched skills")

            if matched_skills:
                for skill in matched_skills:
                    st.write(f"✅ {skill}")
            else:
                st.write("No matched skills detected.")

    with col2:
            st.subheader("Missing Skills")
            st.write(f"{len(missing_skills)} missing skills")

            if missing_skills:
                for skill in missing_skills:
                    st.write(f"⚠️ {skill}")
            else:
                st.write("No major missing skills detected.")

    st.subheader("Recommendation")

    if score_percentage >= 70:
            st.info(
                "Your CV is a strong match for this job description."
            )
    elif score_percentage >= 40:
            st.info(
                "Your CV is a moderate match. Add the missing skills where they genuinely apply."
            )
    else:
            st.info(
                "Your CV has a low match. Add relevant technical skills, tools, and project experience that match the job description."
            ) 

else:
        st.warning(
            "Please upload a CV and paste a job description."
        )
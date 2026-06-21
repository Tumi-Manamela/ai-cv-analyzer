import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI CV Analyzer", page_icon="📄")

st.title("📄 AI CV Analyzer")
st.write("Upload your CV text and compare it to a job description.")

cv_text = st.text_area("Paste your CV here")

job_description = st.text_area("Paste the Job Description here")

if st.button("Analyze CV"):

    if cv_text and job_description:

        documents = [cv_text, job_description]

        tfidf = TfidfVectorizer()

        matrix = tfidf.fit_transform(documents)

        similarity_score = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]

        st.subheader("Results")

        st.success(
            f"Match Score: {round(similarity_score * 100, 2)}%"
        )

        cv_words = set(cv_text.lower().split())
        jd_words = set(job_description.lower().split())

        missing_skills = jd_words - cv_words

        st.subheader("Suggested Keywords to Add")

        for skill in list(missing_skills)[:15]:
            st.write(f"• {skill}")

    else:
        st.warning("Please enter both CV text and Job Description.")

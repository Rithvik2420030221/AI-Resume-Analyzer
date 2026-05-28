from flask import Flask, render_template, request
import PyPDF2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Extract text from PDF
def extract_text(file):
    pdf = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# Calculate similarity
def get_similarity(resume, job_desc):
    cv = CountVectorizer()
    matrix = cv.fit_transform([resume, job_desc])
    similarity = cosine_similarity(matrix)[0][1]
    return round(similarity * 100, 2)

# Extract skills
def extract_skills(text):
    skills_list = ["python", "java", "sql", "machine learning", "data analysis", "html", "css"]
    found_skills = []

    for skill in skills_list:
        if skill in text.lower():
            found_skills.append(skill)

    return found_skills

# Find missing skills
def missing_skills(resume, job_desc):
    skills_list = ["python", "java", "sql", "machine learning", "data analysis", "html", "css"]

    resume_skills = [s for s in skills_list if s in resume.lower()]
    job_skills = [s for s in skills_list if s in job_desc.lower()]

    missing = [s for s in job_skills if s not in resume_skills]

    return missing

# Feedback based on score
def get_feedback(score):
    if score >= 75:
        return "Excellent match! Your resume is strong."
    elif score >= 50:
        return "Good match, but you can improve."
    else:
        return "Low match. Add more relevant skills."

@app.route("/", methods=["GET", "POST"])
def index():
    score = None
    skills = []
    missing = []
    feedback = ""

    if request.method == "POST":
        file = request.files["resume"]
        job_desc = request.form["job_desc"]

        resume_text = extract_text(file)

        score = get_similarity(resume_text, job_desc)
        skills = extract_skills(resume_text)
        missing = missing_skills(resume_text, job_desc)
        feedback = get_feedback(score)

    return render_template("index.html", score=score, skills=skills, missing=missing, feedback=feedback)

if __name__ == "__main__":
    app.run(debug=True)
# 🎓 AI Placement Intelligence Platform
**An End-to-End Career Analytics Tool for Students**

## 📌 Overview
This project addresses low placement rates by providing students with an AI-driven "Gap Analysis." Users upload their resumes to receive a placement probability score, a visualization of their skill gaps compared to successfully placed peers, and a 90-day actionable roadmap.

## 🚀 Live Demo
[🔗 Click here to access the live app on Streamlit Cloud](YOUR_STREAMLIT_URL_HERE)

## 🛠️ Tech Stack
- **Frontend/Deployment:** Streamlit (Python)
- **Machine Learning:** Random Forest Classifier (Scikit-Learn)
- **NLP:** Resume Parsing via PyMuPDF & Regex
- **Analytics:** Plotly, Pandas, NumPy
- **Database:** SQLite (SQLAlchemy)

## 📊 Key Features
1. **Resume Parser:** Extracts skills and education signals directly from PDF/DOCX.
2. **Placement Predictor:** Uses a model trained on 200+ student profiles to estimate job readiness.
3. **Skill Gap Radar:** Compares user skills vs. top 25% salary earners in the dataset.
4. **Actionable Roadmap:** Generates a personalized 30/60/90 day plan based on missing "High Impact" skills.

## 📈 Insights Found
- **Top Placement Predictors:** Python (0.32), Statistics (0.28), and Internship Experience (0.21).
- **Salary Trends:** Students with 3+ core technical skills saw a 40% increase in starting offer values.

## 💻 How to Run Locally
1. Clone the repo: `git clone <your-repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

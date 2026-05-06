import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, re
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="PlaceMe AI", page_icon="🎯", layout="wide")

# ── Load model & data ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("placement_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("students.csv")

model = load_model()
df    = load_data()
SKILLS = json.load(open("skills_list.json"))

# ── Resume parser (keyword match, no spaCy needed) ────────────
def parse_resume(file_bytes, filename):
    try:
        if filename.endswith(".pdf"):
            import fitz  # PyMuPDF
            doc  = fitz.open(stream=file_bytes, filetype="pdf")
            text = " ".join(p.get_text() for p in doc)
        else:
            import docx
            from io import BytesIO
            doc  = docx.Document(BytesIO(file_bytes))
            text = " ".join(p.text for p in doc.paragraphs)
        return text.lower()
    except Exception:
        return ""

def extract_skills(text):
    found = {}
    for sk in SKILLS:
        pattern = sk.replace("_", r"[\s_-]?")
        found[sk] = 1 if re.search(pattern, text) else 0
    return found

def gpa_from_text(text):
    m = re.search(r'\b([34]\.\d{1,2})\b', text)
    return float(m.group(1)) if m else 3.0

# ── Roadmap generator ─────────────────────────────────────────
RESOURCES = {
    "python":          ("Python for Everybody",     "https://www.coursera.org/specializations/python"),
    "sql":             ("SQL for Data Science",      "https://www.coursera.org/learn/sql-for-data-science"),
    "machine_learning":("ML by Andrew Ng",           "https://www.coursera.org/specializations/machine-learning-introduction"),
    "statistics":      ("Statistics with Python",    "https://www.coursera.org/specializations/statistics-with-python"),
    "data_viz":        ("Data Viz with Python",      "https://www.coursera.org/learn/python-for-data-visualization"),
    "deep_learning":   ("Deep Learning Specialisation","https://www.coursera.org/specializations/deep-learning"),
    "nlp":             ("NLP Specialisation",        "https://www.coursera.org/specializations/natural-language-processing"),
    "cloud":           ("Google Cloud Fundamentals", "https://www.coursera.org/learn/gcp-fundamentals"),
    "git":             ("Git & GitHub Crash Course",  "https://www.udemy.com/course/git-and-github-bootcamp/"),
    "tableau":         ("Tableau for Beginners",     "https://www.coursera.org/learn/analytics-tableau"),
    "power_bi":        ("Power BI Masterclass",      "https://www.udemy.com/course/powerbi-complete-introduction/"),
    "excel":           ("Excel Skills for Business", "https://www.coursera.org/specializations/excel"),
    "communication":   ("Business Communication",    "https://www.coursera.org/learn/wharton-communication-skills"),
    "problem_solving": ("Critical Thinking & Problem Solving","https://www.coursera.org/learn/critical-thinking-skills"),
    "teamwork":        ("Teamwork Skills",            "https://www.coursera.org/learn/teamwork-skills"),
}

def build_roadmap(gaps):
    if not gaps:
        return []
    plan = []
    for i, sk in enumerate(gaps[:6]):
        phase = "30-day" if i < 2 else "60-day" if i < 4 else "90-day"
        course, url = RESOURCES.get(sk, (f"Learn {sk}", "#"))
        plan.append({"Phase": phase, "Skill": sk.replace("_"," ").title(),
                     "Action": f"Complete: {course}", "Link": url})
    return plan

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
st.sidebar.title("🎯 PlaceMe AI")
st.sidebar.caption("Student Placement Intelligence")
page = st.sidebar.radio("Navigate", ["📄 Resume Analyser", "📊 Analytics Dashboard", "🏆 Peer Benchmark"])
st.sidebar.markdown("---")
st.sidebar.caption("Built with Python · Streamlit · RandomForest\nDeploy free → streamlit.io")

# ══════════════════════════════════════════════════════════════
# PAGE 1 — RESUME ANALYSER
# ══════════════════════════════════════════════════════════════
if page == "📄 Resume Analyser":
    st.title("📄 Resume Analyser")
    st.caption("Upload your resume and get instant placement insights + a personalised learning roadmap.")

    uploaded = st.file_uploader("Upload resume (PDF or DOCX)", type=["pdf","docx"])

    # ── Manual skill toggle (works without resume too) ────────
    st.subheader("Or enter your skills manually")
    cols = st.columns(5)
    manual = {}
    for i, sk in enumerate(SKILLS):
        manual[sk] = int(cols[i % 5].checkbox(sk.replace("_"," ").title(), key=sk))

    gpa_input = st.slider("Your GPA", 2.0, 4.0, 3.0, 0.1)

    if uploaded or any(manual.values()):
        if uploaded:
            raw   = uploaded.read()
            text  = parse_resume(raw, uploaded.name)
            skills_detected = extract_skills(text)
            gpa   = gpa_from_text(text) if gpa_from_text(text) != 3.0 else gpa_input
            # merge manual overrides
            for sk in SKILLS:
                if manual[sk]: skills_detected[sk] = 1
        else:
            skills_detected = manual
            gpa = gpa_input

        # ── Prediction ────────────────────────────────────────
        X = pd.DataFrame([{**skills_detected, "gpa": gpa}])
        prob     = model.predict_proba(X)[0][1]
        placed   = prob >= 0.5
        skill_count = sum(skills_detected[s] for s in SKILLS)

        # ── Top metrics ───────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.metric("Placement Probability", f"{prob*100:.0f}%",
                  "✅ High" if prob >= 0.7 else "⚠️ Medium" if prob >= 0.4 else "❌ Low")
        c2.metric("Skills Detected", f"{skill_count} / {len(SKILLS)}")
        avg_placed_skills = df[df['placed']==1][SKILLS].sum(axis=1).mean()
        c3.metric("Avg skills of placed students", f"{avg_placed_skills:.0f}")

        st.markdown("---")

        # ── Skill radar ───────────────────────────────────────
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Your skill profile")
            your_vals  = [skills_detected.get(s,0) for s in SKILLS]
            avg_placed = df[df['placed']==1][SKILLS].mean().tolist()
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=your_vals,   theta=[s.replace("_"," ").title() for s in SKILLS], fill='toself', name='You',         line_color='#7F77DD'))
            fig.add_trace(go.Scatterpolar(r=avg_placed,  theta=[s.replace("_"," ").title() for s in SKILLS], fill='toself', name='Placed avg', line_color='#1D9E75', opacity=0.5))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True, height=380, margin=dict(l=40,r=40,t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Skill gap (vs placed students)")
            gaps_vals = {s: avg_placed[i] - skills_detected.get(s,0) for i,s in enumerate(SKILLS)}
            gaps_sorted = sorted(gaps_vals.items(), key=lambda x: x[1], reverse=True)
            gap_df = pd.DataFrame(gaps_sorted, columns=["Skill","Gap"])
            gap_df["Skill"] = gap_df["Skill"].str.replace("_"," ").str.title()
            gap_df["colour"] = gap_df["Gap"].apply(lambda x: "#E24B4A" if x > 0.3 else "#EF9F27" if x > 0 else "#1D9E75")
            fig2 = px.bar(gap_df, x="Gap", y="Skill", orientation="h",
                          color="colour", color_discrete_map="identity",
                          height=380)
            fig2.update_layout(showlegend=False, margin=dict(l=10,r=10,t=20,b=20),
                               xaxis_title="Gap score", yaxis_title="")
            st.plotly_chart(fig2, use_container_width=True)

        # ── 90-day roadmap ────────────────────────────────────
        st.subheader("🗺️ Your personalised 90-day learning roadmap")
        top_gaps = [s for s,g in gaps_sorted if g > 0]
        roadmap  = build_roadmap(top_gaps)
        if roadmap:
            rm_df = pd.DataFrame(roadmap)
            for phase, colour in [("30-day","#EEEDFE"), ("60-day","#E1F5EE"), ("90-day","#E6F1FB")]:
                phase_df = rm_df[rm_df["Phase"] == phase]
                if not phase_df.empty:
                    st.markdown(f"**{phase} goals**")
                    for _, row in phase_df.iterrows():
                        st.markdown(f"- **{row['Skill']}** — [{row['Action']}]({row['Link']})")
        else:
            st.success("🎉 You have all key skills! Focus on projects and interview prep.")

        # ── Salary estimate ───────────────────────────────────
        if placed:
            placed_df = df[df['placed']==1]
            est_salary = int(placed_df['salary'].mean() * (0.8 + prob*0.4))
            st.info(f"💰 Estimated starting salary range: **£{est_salary-3000:,} – £{est_salary+3000:,}**")

# ══════════════════════════════════════════════════════════════
# PAGE 2 — ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════
elif page == "📊 Analytics Dashboard":
    st.title("📊 Placement Analytics Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", len(df))
    c2.metric("Placed", f"{df['placed'].sum()} ({df['placed'].mean()*100:.0f}%)")
    c3.metric("Avg Salary (placed)", f"£{df[df['placed']==1]['salary'].mean():,.0f}")
    c4.metric("Top skill", "Python")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Skill vs placement rate")
        skill_place = pd.DataFrame({
            "Skill": [s.replace("_"," ").title() for s in SKILLS],
            "Placement rate": [df[df[s]==1]['placed'].mean() for s in SKILLS]
        }).sort_values("Placement rate", ascending=False)
        fig = px.bar(skill_place, x="Placement rate", y="Skill", orientation="h",
                     color="Placement rate", color_continuous_scale="teal", height=450)
        fig.update_layout(coloraxis_showscale=False, margin=dict(l=10,r=10,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Salary distribution by course")
        sal_df = df[df['placed']==1]
        fig2 = px.box(sal_df, x="course", y="salary", color="course",
                      color_discrete_sequence=["#7F77DD","#1D9E75","#378ADD","#EF9F27"],
                      height=450)
        fig2.update_layout(showlegend=False, margin=dict(l=10,r=10,t=20,b=20),
                           xaxis_title="", yaxis_title="Salary (£)")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Placement rate by number of skills held")
    df['skill_count'] = df[SKILLS].sum(axis=1)
    skill_grp = df.groupby('skill_count')['placed'].mean().reset_index()
    skill_grp.columns = ['Skills held','Placement rate']
    fig3 = px.line(skill_grp, x='Skills held', y='Placement rate', markers=True,
                   color_discrete_sequence=['#7F77DD'])
    fig3.update_layout(height=300, margin=dict(l=10,r=10,t=20,b=20))
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 3 — PEER BENCHMARK
# ══════════════════════════════════════════════════════════════
elif page == "🏆 Peer Benchmark":
    st.title("🏆 Where Do You Stand?")
    st.caption("Compare yourself against successfully placed students.")

    st.subheader("Select your skills")
    cols = st.columns(5)
    user_skills = {}
    for i, sk in enumerate(SKILLS):
        user_skills[sk] = int(cols[i % 5].checkbox(sk.replace("_"," ").title(), key=f"bm_{sk}"))
    user_gpa = st.slider("Your GPA", 2.0, 4.0, 3.0, 0.1)

    if st.button("Compare me →"):
        user_score  = sum(user_skills.values()) + user_gpa * 0.5
        placed_df   = df[df['placed']==1].copy()
        placed_df['score'] = placed_df[SKILLS].sum(axis=1) + placed_df['gpa'] * 0.5
        percentile  = (placed_df['score'] < user_score).mean() * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Your score",      f"{user_score:.1f}")
        c2.metric("Median (placed)", f"{placed_df['score'].median():.1f}")
        c3.metric("Your percentile", f"Top {100-percentile:.0f}%")

        fig = px.histogram(placed_df, x='score', nbins=20,
                           color_discrete_sequence=['#1D9E75'],
                           labels={'score':'Score'}, title="Your score vs placed students")
        fig.add_vline(x=user_score, line_dash="dash", line_color="#E24B4A",
                      annotation_text="You", annotation_position="top right")
        fig.update_layout(height=350, margin=dict(l=10,r=10,t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)

        missing = [s for s in SKILLS if user_skills[s]==0 and placed_df[s].mean() > 0.5]
        if missing:
            st.warning(f"**Skills most placed students have that you don't:** {', '.join(s.replace('_',' ').title() for s in missing[:5])}")

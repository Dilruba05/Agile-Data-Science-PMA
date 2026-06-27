
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ----------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------

st.set_page_config(
    page_title="Student Course Completion Dashboard",
    page_icon="🎓",
    layout="wide"
)

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "online_course_engagement_data.csv")

rf_model = joblib.load(BASE_DIR / "random_forest_model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
label_encoder = joblib.load(BASE_DIR / "label_encoder.pkl")

# ----------------------------------------------------------
# PREPROCESS DATA FOR DASHBOARD
# ----------------------------------------------------------

# Keep original category names for display
df["CourseCategory_Name"] = df["CourseCategory"]

# Encode course category
df["CourseCategory"] = label_encoder.transform(df["CourseCategory"])

# Create scaled features
scaled = scaler.transform(df[[
    "TimeSpentOnCourse",
    "NumberOfVideosWatched",
    "NumberOfQuizzesTaken",
    "QuizScores",
    "CompletionRate"
]])

df["TimeSpentOnCourse_Scaled"] = scaled[:,0]
df["NumberOfVideosWatched_Scaled"] = scaled[:,1]
df["NumberOfQuizzesTaken_Scaled"] = scaled[:,2]
df["QuizScores_Scaled"] = scaled[:,3]
df["CompletionRate_Scaled"] = scaled[:,4]

# ----------------------------------------------------------
# CUSTOM STYLE
# ----------------------------------------------------------

st.markdown("""
<style>

.main{
    background:#f5f7fb;
}

.metric-card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 2px 8px rgba(0,0,0,0.08);
    text-align:center;
}

h1{
    color:#0d47a1;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------

st.sidebar.title("Dashboard Filters")

course_filter = st.sidebar.selectbox(
    "Course Category",
    ["All"] + sorted(df["CourseCategory_Name"].unique())
)

status_filter = st.sidebar.selectbox(
    "Course Status",
    ["All", "Completed", "Not Completed"]
)


completion_filter = st.sidebar.slider(
    "Completion Rate",
    0,
    100,
    (0,100)
)

search_student = st.sidebar.text_input(
    "Search Student ID"
)

filtered_df = df.copy()

if course_filter != "All":
    filtered_df = filtered_df[
        filtered_df["CourseCategory_Name"] == course_filter
    ]

if status_filter == "Completed":
    filtered_df = filtered_df[
        filtered_df["CourseCompletion"] == 1
    ]

elif status_filter == "Not Completed":
    filtered_df = filtered_df[
        filtered_df["CourseCompletion"] == 0
    ]


filtered_df = filtered_df[
    (filtered_df["CompletionRate"]>=completion_filter[0]) &
    (filtered_df["CompletionRate"]<=completion_filter[1])
]

if search_student != "":
    filtered_df = filtered_df[
        filtered_df["UserID"].astype(str).str.contains(search_student)
    ]


st.title("🎓 Student Course Completion Dashboard")

st.write(
"""
This dashboard provides interactive learning analytics
and predicts student course completion using the
Random Forest model.
"""
)

st.markdown("---")

# ==========================================================
# KPI CALCULATIONS
# ==========================================================

total_students = len(filtered_df)

average_completion = filtered_df["CompletionRate"].mean()

average_quiz = filtered_df["QuizScores"].mean()

high_risk = len(
    filtered_df[
        filtered_df["CompletionRate"] < 33
    ]
)
col1,col2,col3,col4 = st.columns(4)
with col1:

    st.markdown(f"""
    <div class="metric-card">

    <h4 style="color:#1565C0;">
    👨‍🎓 Total Students
    </h4>

    <h1>{total_students:,}</h1>

    <p>100% of dataset</p>

    </div>
    """,
    unsafe_allow_html=True)

with col2:

    st.markdown(f"""
    <div class="metric-card">

    <h4 style="color:#2E7D32;">
    ✅ Average Completion Rate
    </h4>

    <h1>{average_completion:.2f}%</h1>

    <p>Mean completion rate</p>

    </div>
    """,
    unsafe_allow_html=True)

with col3:

    st.markdown(f"""
    <div class="metric-card">

    <h4 style="color:#F57C00;">
    ⭐ Average Quiz Score
    </h4>

    <h1>{average_quiz:.2f}</h1>

    <p>Mean quiz score</p>

    </div>
    """,
    unsafe_allow_html=True)

with col4:

    st.markdown(f"""
    <div class="metric-card">

    <h4 style="color:#D32F2F;">
    🚨 High Risk Students
    </h4>

    <h1>{high_risk:,}</h1>

    <p>Completion Rate below 33%</p>

    </div>
    """,
    unsafe_allow_html=True)
# ==========================================================
# VISUALIZATIONS
# ==========================================================

st.markdown("---")

col_chart1, col_chart2, col_chart3 = st.columns(3)

with col_chart1:

    st.subheader("1. Course Completion Distribution")

    completion_counts = (
        filtered_df["CourseCompletion"]
        .value_counts()
        .rename(index={1: "Completed", 0: "Not Completed"})
    )

    fig1 = px.bar(
        x=completion_counts.index,
        y=completion_counts.values,
        color=completion_counts.index,
        color_discrete_map={
            "Completed": "green",
            "Not Completed": "red"
        },
        labels={"x": "", "y": "Number of Students"}
    )

    fig1.update_layout(
        showlegend=False,
        height=350
    )

    st.plotly_chart(fig1, use_container_width=True)

with col_chart2:

    st.subheader("2. Correlation Heatmap")

    corr = filtered_df[
        [
            "TimeSpentOnCourse",
            "NumberOfVideosWatched",
            "NumberOfQuizzesTaken",
            "QuizScores",
            "CompletionRate",
            "CourseCompletion"
        ]
    ].corr()

    fig2 = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )

    fig2.update_layout(
        height=350
    )

    st.plotly_chart(fig2, use_container_width=True)

with col_chart3:

    st.subheader("3. Feature Importance")

    features = [
    "Course Category",
    "Time Spent",
    "Videos Watched",
    "Quizzes Taken",
    "Quiz Scores",
    "Completion Rate"
]

    importance = rf_model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=True
    )

    fig3 = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues"
    )

    fig3.update_layout(
        coloraxis_showscale=False,
        height=350
    )

    st.plotly_chart(fig3, use_container_width=True)
# ==========================================================
# PREDICTION SECTION
# ==========================================================

st.markdown("---")

st.subheader("4. Predict Course Completion")

left, right = st.columns(2)

with left:

    course = st.selectbox(
        "Course Category",
        ["Arts","Business","Health","Programming","Science"]
    )

    time_spent = st.slider(
        "Time Spent on Course (Hours)",
        1,
        100,
        50
    )

    videos = st.slider(
        "Videos Watched",
        0,
        20,
        10
    )

    quizzes = st.slider(
        "Quizzes Taken",
        0,
        10,
        5
    )

with right:

    quiz_score = st.slider(
        "Quiz Score",
        50,
        100,
        75
    )

    completion_rate = st.slider(
        "Completion Rate (%)",
        0,
        100,
        60
    )

    predict = st.button(
        "Predict Course Completion"
    )

if predict:

    course_encoded = label_encoder.transform([course])[0]

    scaled = scaler.transform([[
        time_spent,
        videos,
        quizzes,
        quiz_score,
        completion_rate
    ]])

    prediction_data = pd.DataFrame({

        "CourseCategory":[course_encoded],

        "TimeSpentOnCourse_Scaled":[scaled[0][0]],

        "NumberOfVideosWatched_Scaled":[scaled[0][1]],

        "NumberOfQuizzesTaken_Scaled":[scaled[0][2]],

        "QuizScores_Scaled":[scaled[0][3]],

        "CompletionRate_Scaled":[scaled[0][4]]

    })

    prediction = rf_model.predict(prediction_data)[0]

    probability = rf_model.predict_proba(prediction_data)[0]

    st.markdown("---")

    if prediction == 1:

        st.success("Prediction : Completed")

        st.metric(
            "Completion Probability",
            f"{probability[1]*100:.2f}%"
        )

        st.info(
            "Recommendation: Continue the current learning behaviour."
        )

    else:

        st.error("Prediction : Not Completed")

        st.metric(
            "Completion Probability",
            f"{probability[1]*100:.2f}%"
        )

        st.warning(
            "Recommendation: Increase quizzes, videos watched and completion rate."
        )
# ==========================================================
# TOP AT-RISK STUDENTS
# ==========================================================

st.markdown("---")

st.subheader("5. Top At-Risk Students")

risk_df = filtered_df.copy()

# Calculate probability of course completion
X_risk = risk_df[[
    "CourseCategory",
    "TimeSpentOnCourse_Scaled",
    "NumberOfVideosWatched_Scaled",
    "NumberOfQuizzesTaken_Scaled",
    "QuizScores_Scaled",
    "CompletionRate_Scaled"
]]

risk_df["Completion_Probability"] = rf_model.predict_proba(X_risk)[:,1]

risk_df["Risk_Probability"] = 1 - risk_df["Completion_Probability"]

risk_df = risk_df.sort_values(
    "Risk_Probability",
    ascending=False
)

top_risk = risk_df.head(5).copy()

def recommendation(rate):

    if rate < 20:
        return "Advisor Meeting"

    elif rate < 40:
        return "Send Learning Resources"

    elif rate < 60:
        return "Personalised Follow-up"

    else:
        return "Weekly Monitoring"

top_risk["Recommended Action"] = (
    top_risk["CompletionRate"]
    .apply(recommendation)
)

display_table = top_risk[[
    "UserID",
    "CompletionRate",
    "Risk_Probability",
    "Recommended Action"
]]

display_table.columns = [
    "Student ID",
    "Completion Rate (%)",
    "Risk Probability",
    "Recommended Action"
]

st.dataframe(
    display_table,
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

st.success(
"""
Dashboard successfully loaded.

Random Forest Accuracy : 95.45%

Supports stakeholder decision-making by combining
interactive analytics with machine learning predictions.
"""
)


  

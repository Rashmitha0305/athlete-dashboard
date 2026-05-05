import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="Athlete Dashboard", layout="wide")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("athlete_data.csv")

# Load ML model
model = pickle.load(open("model.pkl", "rb"))

# -----------------------------
# TITLE
# -----------------------------
st.title("🏆 Athlete Performance Optimization Dashboard")

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("Filters")

sport = st.sidebar.selectbox("Select Sport", df["Sport"].unique())
athlete = st.sidebar.selectbox("Select Athlete", df["Athlete_Name"].unique())

# -----------------------------
# FILTERED DATA
# -----------------------------
filtered_df = df[(df["Sport"] == sport) & (df["Athlete_Name"] == athlete)]

# For insights (sport-level)
sport_df = df[df["Sport"] == sport]

# -----------------------------
# KPIs
# -----------------------------
st.subheader("📊 KPIs")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Performance", int(filtered_df["Performance_Score"].values[0]))
col2.metric("Fitness", int(filtered_df["Fitness_Level"].values[0]))
col3.metric("Efficiency", round(filtered_df["Efficiency_Score"].values[0], 2))
col4.metric("Consistency", round(filtered_df["Consistency_Score"].values[0], 2))

# -----------------------------
# CHARTS (FILTERED)
# -----------------------------
st.subheader("📈 Visual Analysis")

colA, colB = st.columns(2)

# Bar Chart
with colA:
    fig1 = px.bar(
        sport_df,
        x="Athlete_Name",
        y="Performance_Score",
        color="Athlete_Name",
        title=f"{sport} - Athlete Performance Comparison"
    )
    st.plotly_chart(fig1, use_container_width=True)

# Scatter Plot
with colB:
    fig2 = px.scatter(
        sport_df,
        x="Training_Hours",
        y="Performance_Score",
        color="Athlete_Name",
        title="Training vs Performance"
    )
    st.plotly_chart(fig2, use_container_width=True)

# Box Plot
fig3 = px.box(
    sport_df,
    x="Injury_Count",
    y="Performance_Score",
    color="Athlete_Name",
    title="Injury Impact Analysis"
)
st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# ML PREDICTION
# -----------------------------
st.subheader("🤖 Performance Prediction")

colP1, colP2 = st.columns(2)

with colP1:
    speed = st.slider("Speed", 60, 100, 80)
    stamina = st.slider("Stamina", 60, 100, 80)

with colP2:
    fitness = st.slider("Fitness Level", 60, 100, 85)
    training = st.slider("Training Hours", 5, 25, 15)

input_data = pd.DataFrame(
    [[speed, stamina, fitness, training]],
    columns=["Speed", "Stamina", "Fitness_Level", "Training_Hours"]
)

pred = model.predict(input_data)

st.success(f"Predicted Performance Score: {round(pred[0],2)}")

# -----------------------------
# INSIGHTS (UPDATED)
# -----------------------------
st.subheader("📌 Insights")

# Top performer in selected sport
top_player = sport_df.loc[sport_df["Performance_Score"].idxmax()]["Athlete_Name"]
st.write(f"🏆 Top Performer in {sport}: **{top_player}**")

# Bonus insight (average performance)
avg_perf = sport_df["Performance_Score"].mean()

if avg_perf > 85:
    st.write("📈 This sport shows high overall performance levels")
else:
    st.write("📊 Performance varies significantly in this sport")

st.write("📈 Higher training hours generally improve performance")
st.write("⚠️ Increased injuries reduce performance consistency")

# -----------------------------
# RECOMMENDATIONS (UPDATED)
# -----------------------------
st.subheader("💡 Recommendations")

injury = filtered_df["Injury_Count"].values[0]
fitness_val = filtered_df["Fitness_Level"].values[0]
eff = filtered_df["Efficiency_Score"].values[0]

# Injury-based recommendation
if injury > 3:
    st.warning("⚠️ High injury risk – recommend rest and recovery")
elif injury > 1:
    st.info("Moderate injury risk – monitor workload")
else:
    st.success("Low injury risk – athlete is in good condition")

# Fitness-based recommendation
if fitness_val < 75:
    st.warning("Improve fitness with endurance and strength training")
else:
    st.success("Fitness level is good")

# Efficiency-based recommendation
if eff < 5:
    st.info("Optimize training strategy to improve efficiency")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Developed using Streamlit | Sports Data Analytics Project")
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Week Schedule", layout="wide")

########################################################################################################
# --- Top banner ---
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 3rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div style="background-color:#002D62; padding:4px 10px; border-radius:10px; margin-bottom:10px;">
        <h1 style="color:white; text-align:center;">🏈 BG.Analytics CFB Dashboard</h1>
    </div>
    """,
    unsafe_allow_html=True
)
########################################################################################################

# ---- Load Data ----
@st.cache_data
def load_data():
    df = pd.read_csv("data/game_data_current.csv")
    df["startDate"] = pd.to_datetime(df["startDate"])
    df["startDate"] = pd.to_datetime(df["startDate"]).dt.strftime("%Y-%m-%d")
    return df

df = load_data()

# CUSTOM COLUMN CREATIONS

# Score
def format_score(row):
    if pd.isna(row["teamPoints"]) or pd.isna(row["opponentPoints"]):
        return "TBD"
    return f"{int(row['teamPoints'])}–{int(row['opponentPoints'])}"

df["Score"] = df.apply(format_score, axis=1)

# CFP Importance (80% avg, 20% closeness)
epsilon = 0.01
group = df.groupby("id")
avg_points = group["potentialPlayoffPoints"].transform("mean")
diff = group["potentialPlayoffPoints"].transform(lambda x: abs(x.iloc[0] - x.iloc[1]) if len(x) == 2 else 0)
closeness_score = 1 / (diff + epsilon)
df["CFP Importance"] = (0.97 * avg_points + 0.03 * closeness_score).round(2)

# ---- Find Upcoming Week ----
incomplete_games = df[~df["completed"]]
upcoming_week = incomplete_games["week"].min()

# ---- Week Selector ----
week_list = sorted(df["week"].dropna().unique())
selected_week = st.selectbox("📅 Select Week", options=week_list, index=week_list.index(upcoming_week))

# ---- Filter to Selected Week ----

week_df = df[df["week"] == selected_week].copy()
# ---- Deduplicate to one row per game (prefer Home perspective) ----
pref_map = {"home": 0, "neutral": 1, "away": 2}
week_df["_pref"] = (
    week_df["location"].astype(str).str.strip().str.lower().map(pref_map).fillna(3)
)
# Keep the preferred row (Home > Neutral > Away) within each game id
week_df = (
    week_df.sort_values(["id", "_pref", "startDate", "team"])  # stable sort
           .groupby("id", as_index=False)
           .first()
)
week_df.drop(columns=["_pref"], inplace=True, errors="ignore")

# ---- Define Display Columns ----
# 👉 Customize this list freely
# ---- Define Display Columns (freely customize) ----
week_df.rename(columns={"startDate": "Date", "startTime": "Time", "team": "Team1", "opponent": "Team2"}, inplace=True)
display_columns = [
    "Date", "Time", "Team1", "Team2", "CFP Importance"
]

# ---- Define Rankable Columns ----
rankable_columns = ["CFP Importance"]

# ---- Sort By Dropdown ----
sort_column = st.selectbox("📊 Sort By", options=rankable_columns, index=rankable_columns.index("CFP Importance"))

# ---- Add Rank ----
if sort_column in week_df.columns:
    week_df["Rank"] = week_df[sort_column].rank(ascending=False, method="min").astype(int)

# ---- Final Display ----
final_columns = ["Rank"] + display_columns
week_df = week_df[final_columns].sort_values("Rank")

# ---- Optional Styling ----
numeric_cols = ["CFP Importance"]
styled = week_df.style.format({
    "CFP Importance": "{:.2f}"
}).background_gradient(subset=numeric_cols, cmap="RdYlGn")
# ---- Show Table ----
st.markdown(f"### 🏈 Schedule for Week {selected_week}")
st.dataframe(styled, use_container_width=False, width=None, hide_index=True, height=563)
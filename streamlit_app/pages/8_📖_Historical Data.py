import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Historical Data", layout="wide")

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

st.title("📚 Historical Team Data")

# ---- Load Data ----
@st.cache_data
def load_data():
    game_data = pd.read_csv("data/game_data.csv")
    return game_data

game_data = load_data()

# Ensure correct dtypes
game_data["season"] = game_data["season"].astype(int)
game_data["conferenceGame"] = game_data["conferenceGame"].astype(str)
game_data["location"] = game_data["location"].fillna("Unknown")

# ---- MAIN FILTERS (TOP OF SCREEN) ----
st.markdown("### 📅 Filter Historical Data")

years = sorted(game_data["season"].unique())

col1, col2, col3, col4 = st.columns(4)

with col1:
    start_year, end_year = st.select_slider(
        "Season Range",
        options=years,
        value=(years[-1], years[-1])
    )

with col2:
    location_filter = st.selectbox("Location", options=["All", "Home", "Away"])

with col3:
    conference_filter = st.selectbox("Conference Game", options=["All", "True", "False"])

with col4:
    season_type_filter = st.selectbox("Season Type", options=["All"] + sorted(game_data["seasonType"].dropna().unique()), index=2)

# ---- Apply Filters ----
filtered = game_data[
    (game_data["season"] >= start_year) &
    (game_data["season"] <= end_year)
]

if location_filter != "All":
    filtered = filtered[filtered["location"] == location_filter]

if conference_filter != "All":
    filtered = filtered[filtered["conferenceGame"] == conference_filter]

if season_type_filter != "All":
    filtered = filtered[filtered["seasonType"] == season_type_filter]

# ---- Aggregate by Team ----
team_summary = (
    filtered[filtered["completed"] == True]
    .groupby("team")
    .agg(
        Games=("team", "count"),
        Wins=("result", lambda x: (x == "Win").sum()),
        Losses=("result", lambda x: (x == "Loss").sum()),
        Avg_MOV=("MOV", "mean"),
        Total_Playoff_Points=("actualPlayoffPoints", "sum")
    )
    .sort_values("Total_Playoff_Points", ascending=False)
    .reset_index()
)

# ---- Sort Selector ----
sort_options = ["Total_Playoff_Points", "Avg_MOV", "Wins", "Games"]
sort_column = st.selectbox("🔽 Sort by Column", options=sort_options)

# ---- Apply Sort and Ranking ----
team_summary["Rank"] = team_summary[sort_column].rank(method="min", ascending=False).astype(int)
team_summary = team_summary.sort_values(sort_column, ascending=False)
cols = ["Rank"] + [col for col in team_summary.columns if col != "Rank"]
team_summary = team_summary[cols]

# ---- Display Table ----
team_summary = team_summary.rename(columns={
    "Rank": "Rank",
    "team": "Team",
    "Games": "Games Played",
    "Wins": "Wins",
    "Losses": "Losses",
    "Avg_MOV": "Avg MOV",
    "Total_Playoff_Points": "Total Playoff Points"
})

color_scale_columns = [
    "Total Playoff Points",
    "Avg MOV",
    "Wins"
]


st.markdown(f"### 📊 Team Summary: {start_year} to {end_year}")
# ---- Style Table ----
styled_df = team_summary.style.format({
    "Avg MOV": "{:.2f}",
    "Total Playoff Points": "{:.2f}"
}).background_gradient(
    subset=color_scale_columns,
    cmap="RdYlGn"
)

st.dataframe(
    styled_df,
    use_container_width=False,
    width=None,
    hide_index=True,
    height=1000
)
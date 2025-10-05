import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from functions import load_data

st.set_page_config(page_title="College Football Playoff Dashboard", layout="wide")

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

# =============================
# Home: Landing Content
# =============================

# ---------- 1) About BG.Analytics ----------
st.markdown("""
### 👋 About BG.Analytics
I am a College Football Data Scientist who models game results and season-long outcomes for every FBS College Football Team. I specialize in predicting the CFP with my statistical metric "Playoff Points".
Playoff Points give the winner of each matchup a set number of points based on how impressive the win was. This takes in a variety of factors like opponent talent, location, MOV, and a few other key game results. 
Information about the historical predictive performance of Playoff Points can be seen in the "Historical Data" tab of this dashboard.

**Follow Along on Instagram**: [@bg.analytics](https://www.instagram.com/bg.analytics/)
""")

st.divider()

# ---------- 2) What This Page Offers & Tab Guide ----------
st.markdown("""
### 🧭 What you'll find here
This dashboard is organized into a few different sections. Dive right in or explore this guide first:

- **Single Team Outlook** — Individual win probabilties for selected team's games and season outlooks.
- **League Projections** — Ranking team season-long outlooks and comparing these results across the league.
- **Season Simulation** — Run's full season simulations for the entire league and shows conference and playoff probabilities.
- **Playoff Paths** — Analyzes hypothetical situations for various teams and tracks team's paths to the playoffs.
- **Matchup Predictor** — See win probabilites for specific games and project win probabilities for hypothetical future matchups.
- **This Week Schedule** — See the most important games for any given week in the season.
- **Performance Results** — Anlyze the predictive results of our win probability model.
- **Historical Data** — Track the historical results of Playoff Points and see how they predict CFP berths by year.
""")

# ---------- 3) Current Playoff Point Standings (Top 12) ----------
st.markdown("""
### 🏆 Current Playoff Point Standings (Top 12)""")
game_data, season_data, current, map = load_data()

# Group by team and sum Playoff Points
standings = (
    current.groupby("team", as_index=False)
    .agg(
        PlayoffPoints=("actualPlayoffPoints", "sum"),
        Wins=("result", lambda x: (x == "Win").sum()),
        Losses=("result", lambda x: (x == "Loss").sum())
    )
)

# Create a Record column like "6-1"
standings["Record"] = standings["Wins"].astype(str) + "-" + standings["Losses"].astype(str)

# Sort by total playoff points (descending) and take top 12
top12 = standings.sort_values("PlayoffPoints", ascending=False).head(12).reset_index(drop=True)

# Reorder columns for clarity
top12 = top12[["team", "Record", "PlayoffPoints"]]

# Optional: rename columns for display
top12.columns = ["Team", "Record", "Playoff Points"]

# Display
st.dataframe(top12, hide_index=True, use_container_width=True)
st.caption("*Top 12 teams by total Playoff Points. This should contain close to all 12 playoff teams by the end of the season.*")

st.divider()

# ---------- 4) Season-long Prediction Snapshot (Actual Performance Results) ----------
st.markdown("""
### 🔮 Season-long Prediction Results
How well are predictions doing so far this season? We compare **actual outcomes** vs **model win probabilities** across completed games.
""")

# ---- Helper: find the most likely win-probability column ----
candidate_prob_cols = ["winProb", "WinProb", "win_probability", "predWinProb", "prob"]
prob_col = next((c for c in candidate_prob_cols if c in current.columns), None)

if prob_col is None:
    st.warning("Could not find a win probability column in `current` (looked for: "
               + ", ".join(candidate_prob_cols) + "). Please add/rename one and reload.")
else:
    df = current.copy()

    # ---- Completed games filter ----
    # Use 'completed' if present; otherwise infer from a 'result' column or final scores.
    if "completed" in df.columns:
        games = df[df["completed"] == True].copy()
    elif "result" in df.columns:
        games = df[df["result"].isin(["Win", "Loss"])].copy()
    elif {"teamScore", "opponentScore"}.issubset(df.columns):
        games = df[df["teamScore"].notna() & df["opponentScore"].notna()].copy()
    else:
        games = df.copy()  # fallback (may include future games)

    # ---- Create binary outcome (1 = win, 0 = loss) ----
    if "result" in games.columns:
        games["y"] = (games["result"] == "Win").astype(int)
    elif {"teamScore", "opponentScore"}.issubset(games.columns):
        games["y"] = (games["teamScore"] > games["opponentScore"]).astype(int)
    else:
        st.warning("Could not determine outcomes (need `result` or final scores).")
        games["y"] = np.nan

    # Drop rows without probability or outcome
    games = games.dropna(subset=[prob_col, "y"]).copy()

    # Safety: clip probabilities to avoid log(0)
    games["p"] = games[prob_col].astype(float).clip(1e-6, 1 - 1e-6)

    # ---- Threshold-based prediction (0.50) ----
    games["pred_label"] = (games["p"] > 0.5).astype(int)

    # ---- Metrics ----
    n_games = len(games)
    if n_games == 0:
        st.info("No completed games found to evaluate yet.")
    else:
        accuracy = (games["pred_label"] == games["y"]).mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Games Evaluated", f"{n_games}")
        col2.metric("Accuracy", f"{accuracy:.3f}")

        st.markdown("#### 📈 Actual vs Expected Wins by Team (Luckiest Teams)")
        team_perf = (
            games.groupby("team", as_index=False)
                 .agg(
                     Games=("y", "count"),
                     ActualWins=("y", "sum"),
                     ExpectedWins=("p", "sum"),
                 )
        )
        team_perf["OverUnder"] = team_perf["ActualWins"] - team_perf["ExpectedWins"]
        team_perf = team_perf.sort_values(["OverUnder", "ActualWins"], ascending=[False, False])

        # Optional: record string if you also track losses in this same dataset
        # If not, derive losses = Games - ActualWins
        team_perf["Losses"] = (team_perf["Games"] - team_perf["ActualWins"]).astype(int)
        team_perf["Record"] = team_perf["ActualWins"].astype(int).astype(str) + "-" + team_perf["Losses"].astype(str)

        # Tidy columns for display
        perf_display = team_perf[["team", "Record", "Games", "ActualWins", "ExpectedWins", "OverUnder"]]
        perf_display.columns = ["Team", "Record", "Games", "Actual Wins", "Expected Wins", "Over/Under"]

        st.dataframe(perf_display, hide_index=True, use_container_width=True)

        # ---- Optional: per-team drilldown ----
        st.markdown("#### 🔎 Team Drilldown")
        team_list = sorted(games["team"].unique().tolist())
        sel_team = st.selectbox("Choose a team to view game-by-game performance", options=["(All)"] + team_list, index=0)

        if sel_team != "(All)":
            g = games[games["team"] == sel_team].copy()
            # Make a compact per-game table
            # Try to show opponent/date if present
            cols = []
            if "startDate" in g.columns:
                g["Date"] = pd.to_datetime(g["startDate"], errors="coerce").dt.date.astype(str)
                cols.append("Date")
            elif "date" in g.columns:
                g["Date"] = pd.to_datetime(g["date"], errors="coerce").dt.date.astype(str)
                cols.append("Date")

            if "opponent" in g.columns: cols.append("opponent")
            if "location" in g.columns: cols.append("location")
            cols += ["y", "p", "pred_label"]

            show = g[cols].copy()
            show.rename(columns={
                "opponent": "Opponent",
                "location": "Location",
                "y": "Outcome (1=Win)",
                "p": "Win Prob",
                "pred_label": "Pred (≥0.5)"
            }, inplace=True)

            st.dataframe(show, hide_index=True, use_container_width=True)

        st.caption("*Over/Under = Actual Wins − Expected Wins (sum of win probabilities). Positive = outperforming expectations.*")

st.divider()
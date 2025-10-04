import sys
import os
sys.path.append(os.getcwd())
from functions import load_data, simulate_full_season, train_win_model

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go

# =========================
# 🔧 PAGE CONFIGURATION
# =========================
st.set_page_config(page_title="College Football Playoff Dashboard", layout="wide")

st.markdown("""
<style>
    .block-container {
        padding-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# 🌟 DONUT CHART FUNCTIONS
# =========================
def bowl_probability_donut(probability):
    percent = round(probability * 100, 1)
    fig = go.Figure(data=[go.Pie(values=[probability, 1 - probability], hole=0.5, marker_colors=[team_color, '#eee'], sort=False, direction='clockwise', textinfo='none')])
    fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), width=200, height=200, paper_bgcolor="rgba(0,0,0,0)", annotations=[dict(text=f"<b>{percent:.1f}%</b>", x=0.5, y=0.5, font_size=20, showarrow=False)])
    return fig

def playoff_probability_donut(playoff_prob):
    remaining = round(100 - playoff_prob, 1)
    fig = go.Figure(data=[
        go.Pie(
            labels=[f"Make Playoffs ({playoff_prob}%)", f"Miss Playoffs ({remaining}%)"],
            values=[playoff_prob, remaining],
            hole=0.5,
            marker_colors=[team_color, "#eee"],
            textinfo="label"
        )
    ])
    fig.update_layout(
        showlegend=False,
        margin=dict(t=0, b=0, l=0, r=0),
        width=200,
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(text=f"<b>{playoff_prob:.1f}%</b>", x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    return fig

# =========================
# 📊 LOAD & PREPARE DATA
# =========================
game_data, season_data, current, map = load_data()
win_model, win_model_features = train_win_model(game_data)

#Set future game results as "TBD"
current.loc[~current["completed"], "result"] = "TBD"
#Make sure week is read as string
game_data['week'] = game_data['week'].astype(str)
#Set the week for postseason games as "Playoffs"
game_data.loc[game_data['seasonType'] == 'postseason', 'week'] = "Playoffs"
#Set the start time as "TBD" if it is not known yet
current.loc[current['startTimeTBD'].astype(bool), 'startTime'] = "TBD"

# =========================
# 🌟 DASHBOARD HEADER + TEAM SELECTOR + DONUTS
# =========================
teams = sorted(current["team"].unique())
selected_team = "-- Select a team --"
st.markdown("""
    <div style="background-color:#002D62; padding:4px 10px; border-radius:10px; margin-bottom:10px;">
        <h1 style="color:white; text-align:center;">🏈 College Football Playoff Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)

col_left, col_mid, col_right = st.columns([1, 2, 1])

with col_mid:
    selected_team = st.selectbox("Select a Team", ["-- Select a team --"] + teams, key="team_selector")

def simulate_single_team(current_df, team, color):
    team_df = current_df[current_df["team"] == team].copy()
    team_df = team_df.sort_values("startDate")
    team_df["startDate"] = pd.to_datetime(team_df["startDate"])

    COEF_MOV = 0.08237826714816475
    MOV_STD_DEV = 10
    num_simulations = 500
    simulations = []

    for _ in range(num_simulations):
        df = team_df.copy()
        df["actualEarned"] = df.apply(lambda row: row["actualPlayoffPoints"] if row["completed"] and row["result"] == "Win" else 0, axis=1)

        def simulate_proj_points(row):
            if row["completed"]:
                return 0
            win_prob = row.get("winProb", 0.5)
            if np.random.rand() < win_prob:
                expected_mov = row.get("predictedPoints", 0) - row.get("predictedOpponentPoints", 0)
                simulated_mov = np.random.normal(loc=expected_mov, scale=MOV_STD_DEV)
                return row["potentialPlayoffPoints"] + simulated_mov * COEF_MOV
            return 0

        df["projEarned"] = df.apply(simulate_proj_points, axis=1)
        df["cumulativePoints"] = (df["actualEarned"] + df["projEarned"]).cumsum()
        sim_points = [0] + df["cumulativePoints"].tolist()
        sim_dates = [df["startDate"].min() - pd.Timedelta(days=6)] + df["startDate"].tolist()
        simulations.append((sim_dates, sim_points))

    fig = go.Figure()
    for sim_dates, sim_points in simulations:
        fig.add_trace(go.Scatter(x=sim_dates, y=sim_points, mode="lines", line=dict(color=color, width=1), opacity=0.2, showlegend=False))

    fig.add_shape(type="line", x0=team_df["startDate"].min()- pd.Timedelta(days=6), x1=team_df["startDate"].max(), y0=120, y1=120, line=dict(color="red", width=2), layer="below")
    fig.add_annotation(x=team_df["startDate"].min(), y=118, text="Playoff Conversation", showarrow=False, font=dict(color="red", size=12), xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.8)")
    fig.add_shape(type="line", x0=team_df["startDate"].min()- pd.Timedelta(days=6), x1=team_df["startDate"].max(), y0=130, y1=130, line=dict(color="darkgreen", width=2), layer="below")
    fig.add_annotation(x=team_df["startDate"].min(), y=132, text="Playoff Lock", showarrow=False, font=dict(color="darkgreen", size=12), xanchor="left", yanchor="bottom", bgcolor="rgba(255,255,255,0.8)")
    fig.update_layout(title=f"{team} Simulated Seasons (500 Runs)", xaxis_title="Week", yaxis_title="Playoff Points", height=600)

    return fig

if selected_team != "-- Select a team --":
    team_color = map.loc[map["cfb_name"] == selected_team, "Color1"].values[0] if selected_team in map["cfb_name"].values else "#A9A9A9"
    team_current = current[current["team"] == selected_team].copy()
    team_current = team_current.sort_values("startDate")
    wins = (team_current["result"] == "Win").sum()
    upcoming = team_current[~team_current["completed"]].copy()
    upcoming["winProb"] = upcoming["winProb"].fillna(0.5)
    win_probs = pd.to_numeric(upcoming["winProb"], errors="coerce").dropna()

    n_sim = 10000
    sims = np.random.binomial(1, win_probs.values[:, None], size=(len(win_probs), n_sim)) if not win_probs.empty else np.zeros((0, n_sim))
    sim_wins = sims.sum(axis=0) + wins if sims.size else np.zeros(n_sim)
    bowl_odds = (sim_wins >= 6).mean() if sims.size else 0.0

    COEF_MOV = 0.08237826714816475
    MOV_STD_DEV = 10
    playoff_scores = []
    for _ in range(1000):
        df = team_current.copy()
        df["actualEarned"] = df.apply(lambda row: row["actualPlayoffPoints"] if row["completed"] and row["result"] == "Win" else 0, axis=1)
        df["projEarned"] = df.apply(lambda row: row["potentialPlayoffPoints"] + np.random.normal(loc=row["predictedPoints"] - row["predictedOpponentPoints"], scale=MOV_STD_DEV) * COEF_MOV if not row["completed"] and np.random.rand() < row.get("winProb", 0.5) else 0, axis=1)
        df["cumulativePoints"] = (df["actualEarned"] + df["projEarned"]).cumsum()
        final_score = df["cumulativePoints"].iloc[-1]
        playoff_scores.append(1.0 if final_score >= 130 else 0.4 if final_score >= 120 else 0.0)

    playoff_prob = np.mean(playoff_scores)
    sim_dist = pd.Series(sim_wins).value_counts(normalize=True).sort_index()

    with col_left:
        st.markdown("<div style='text-align: center; font-size:20px;'>🎯 <b>Bowl Eligibility Odds</b></div>", unsafe_allow_html=True)
        st.plotly_chart(bowl_probability_donut(bowl_odds), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Playoff Probability logic:
    full_sim_df = simulate_full_season(current, win_model, win_model_features, num_simulations=500)
    team_row = full_sim_df[full_sim_df["team"] == selected_team]
    playoff_prob = team_row["playoff_pct"].values[0] if not team_row.empty else 0.0
    
    with col_right:
        st.markdown("<div style='text-align: center; font-size:20px;'>🏆 <b>CFP Odds</b></div>", unsafe_allow_html=True)

        if playoff_prob is None:
            st.markdown("<div style='text-align: center;'>🔄 <i>Loading projection...</i></div>", unsafe_allow_html=True)
        else:
            st.plotly_chart(playoff_probability_donut(playoff_prob), use_container_width=True)
            st.markdown(
                "<div style='text-align: center; font-size:13px; color:gray;'>*Only 500 simulations for efficiency. See Season Simulation Tab for more Accurate Numbers</div>",
                unsafe_allow_html=True
            )

    with col_mid: 
        team_current = current[current["team"] == selected_team].copy()
        team_current = team_current.sort_values("startDate")
        team_current["startDate"] = pd.to_datetime(team_current["startDate"])

        team_current["actualEarned"] = team_current.apply(lambda row: row["actualPlayoffPoints"] if row["completed"] and row["result"] == "Win" else 0, axis=1)
        team_current["projectedEarned"] = team_current.apply(
            lambda row: (
                row["potentialPlayoffPoints"] + ((row.get("predictedPoints", 0) - row.get("predictedOpponentPoints", 0)) * COEF_MOV)
                if not row["completed"] and row.get("winProb", 0.5) > 0.5
                else 0
            ),
            axis=1
        )
        team_current["cumulativeActual"] = team_current["actualEarned"].cumsum()
        team_current["cumulativeProjected"] = (team_current["actualEarned"] + team_current["projectedEarned"]).cumsum()

        completed = team_current[team_current["completed"]].copy()
        incomplete = team_current[~team_current["completed"]].copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=completed["startDate"], y=completed["cumulativeActual"], mode="lines", name="Actual", line=dict(color=team_color, width=3)))
        if not incomplete.empty:
            fig.add_trace(go.Scatter(x=incomplete["startDate"], y=incomplete["cumulativeProjected"].iloc[len(completed):], mode="lines", name="Projected", line=dict(color=team_color, width=3, dash="dash")))

        fig.update_layout(title="Current and Projected Playoff Points Earned Over Time", xaxis_title="Date", yaxis_title="Cumulative Playoff Points", height=300, showlegend=True)
        
        fig.add_shape(
            type="line",
            x0=team_current["startDate"].min(),
            x1=team_current["startDate"].max(),
            y0=130,
            y1=130,
            line=dict(color="green", width=2, dash="dot")
        )
        fig.add_annotation(
            x=team_current["startDate"].min(),
            y=130,
            text="Playoff Lock",
            showarrow=False,
            font=dict(color="green", size=12),
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.8)"
        )
        
        st.plotly_chart(fig, use_container_width=True)

    current = team_current.copy()

    # Store season sim for later display
    sim_dist = pd.Series(sim_wins).value_counts(normalize=True).sort_index()

    # Save updated team_current for use below
    current = team_current.copy()

Past_Season_Columns = ["startDate", "startTime", "week", "opponent", "location", "result", "Score"]
Current_Season_Columns = ["Date", "startTime", "opponent", "location", "result", "Score", "winProb", 'Score Prediction', "Playoff Points"]

if selected_team != "-- Select a team --":
    left_col, right_col = st.columns([1.2, 1])
    team_color = map.loc[map["cfb_name"] == selected_team, "Color1"].values[0] if selected_team in map["cfb_name"].values else "#A9A9A9"

    with left_col:
        st.subheader(f"{current['season'].max()} Season: All Games for {selected_team}")
        current = current[(current["team"] == selected_team)].sort_values("startDate").copy()

        if not current.empty:
            current["Score"] = current.apply(lambda row: "TBD" if pd.isna(row["teamPoints"]) or pd.isna(row["opponentPoints"]) else f"{int(row['teamPoints'])}–{int(row['opponentPoints'])}", axis=1)
            current["Playoff Points"] = current.apply(lambda row: row.get("actualPlayoffPoints", 0) if row["completed"] else row.get("potentialPlayoffPoints", 0), axis=1)
            current["Date"] = pd.to_datetime(current["startDate"]).dt.strftime("%B %d, %Y")
            current["Score Prediction"] = current.apply(lambda row: "N/A" if pd.isna(row["predictedPoints"]) or pd.isna(row["predictedOpponentPoints"]) else f"{int(round(row['predictedPoints']))}–{int(round(row['predictedOpponentPoints']))}", axis=1)

            styled_df = current[Current_Season_Columns].rename(columns={"winProb": "Win Prob", "startTime": "Time", "opponent": "Opponent", "location": "Location", "result": "Result"}).style.format({"Win Prob": "{:.2%}", "Playoff Points": "{:.2f}"}).background_gradient(subset=["Win Prob"], cmap="RdYlGn", vmin=0.0, vmax=1.0)

            st.dataframe(styled_df, use_container_width=False, width=None, hide_index=True, height=460)
        else:
            st.info(f"No games found for {current['season'].max()}.")

        st.subheader(f"🕰️ {current['season'].max() - 1} Season: Completed Games for {selected_team}")
        game_data = game_data[(game_data["team"] == selected_team) & (game_data["season"] == current['season'].max() - 1) & (game_data["completed"])].sort_values("startDate", ascending=False)

        if not game_data.empty:
            game_data["Score"] = game_data.apply(lambda row: "TBD" if pd.isna(row["teamPoints"]) or pd.isna(row["opponentPoints"]) else f"{int(row['teamPoints'])}–{int(row['opponentPoints'])}", axis=1)
            st.dataframe(game_data[Past_Season_Columns], use_container_width=True, hide_index=True)
        else:
            st.info(f"No completed games found for {current['season'].max() - 1}.")

    with right_col:
        st.subheader(f"🏅 {selected_team} Season Summary")
        wins = (current["result"] == "Win").sum()
        losses = (current["result"] == "Loss").sum()

        current_points = current[(current["completed"]) & (current["result"] == "Win")]["Playoff Points"].sum()
        projected_points = current_points
        incomplete = current[~current["completed"]].copy()
        if not incomplete.empty:
            incomplete["winProb"] = incomplete["winProb"].fillna(0.5)
            incomplete["predictedMOV"] = (incomplete["predictedPoints"] - incomplete["predictedOpponentPoints"]).fillna(0)
            COEF_MOV = 0.08237826714816475
            incomplete["predictedPlayoffPoints"] = incomplete["potentialPlayoffPoints"] + (incomplete["predictedMOV"] * COEF_MOV)
            projected_points += (incomplete["winProb"] * incomplete["predictedPlayoffPoints"]).sum()

            last_game_week = (
                current[(current["team"] == selected_team) & (current["completed"])]
                ["week"]
                .max()
            )

            # rank in the *next* week
            latest_rank = (
                current[
                    (current["team"] == selected_team) & (current["week"] == last_game_week + 1)
                ]
                .sort_values("startDate")
                ["teamRank"]
                .dropna()
                .head(1)
            )

        conf_prob = team_row["conf_champ_pct"].values[0] if not team_row.empty else 0.0
        col1, col2, col3, col4 = st.columns([1, 1, 2, 2])
        col1.metric(label="Record", value=f"{wins}-{losses}")
        col2.metric(label="Rank", value=f"#{int(latest_rank.values[0])}" if not latest_rank.empty else "N/A")
        col3.metric(label="Conf Champion %", value=f"{conf_prob:.2f}%")
        col4.metric(label="Playoff %", value=f"{playoff_prob:.2f}%")

        st.subheader("📈 Simulated Win Total Distribution")
        upcoming = current[~current["completed"]].copy()
        if not upcoming.empty:
            upcoming["winProb"] = upcoming["winProb"].fillna(0.5)
            win_probs = pd.to_numeric(upcoming["winProb"], errors="coerce").dropna()
            if not win_probs.empty:
                n_sim = 10000
                win_probs_array = win_probs.values[:, None]
                sims = np.random.binomial(1, win_probs_array, size=(len(win_probs_array), n_sim))
                sim_wins = sims.sum(axis=0) + wins
                win_dist = pd.Series(sim_wins).value_counts(normalize=True).sort_index()
                st.bar_chart((win_dist * 100).round(2), color=team_color)
            else:
                st.info("No win probabilities available to simulate season.")
        else:
            st.info("No remaining games to simulate.")
        
        st.subheader("🎯 Single Team Season Simulations")
        color = team_color
        sim_chart = simulate_single_team(current, selected_team, color)
        st.plotly_chart(sim_chart, use_container_width=True)
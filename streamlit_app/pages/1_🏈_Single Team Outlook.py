
import sys
import os
sys.path.append(os.getcwd())
from functions import load_data, simulate_full_season, train_win_model

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go

# ==================================================
# 🔧 PAGE CONFIGURATION
# ==================================================
st.set_page_config(page_title="Single Team Analysis", layout="wide")

st.markdown(
    """
    <style>
        .block-container { padding-top: 3rem; }
        .center { text-align:center; }
        .muted { color: gray; font-size: 13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="background-color:#002D62; padding:4px 10px; border-radius:10px; margin-bottom:10px;">
        <h1 style="color:white; text-align:center;">🏈 BG.Analytics CFB Dashboard</h1>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("First run has about a 10 second delay. All others are immediate.")

# ==================================================
# 📦 CONSTANTS
# ==================================================
COEF_MOV: float = 0.08237826714816475
MOV_STD_DEV: int = 10
DEFAULT_COLOR: str = "#A9A9A9"

# ==================================================
# 📊 LOAD DATA & MODEL (one time per rerun)
# ==================================================
# Keep the original load behavior and prep steps
# (This mirrors the original logic but is grouped earlier.)
game_data, season_data, current, map = load_data()
win_model, win_model_features = train_win_model(game_data)

# Normalize a few fields for display consistency
current.loc[~current["completed"], "result"] = "TBD"
game_data["week"] = game_data["week"].astype(str)
game_data.loc[game_data["seasonType"] == "postseason", "week"] = "Playoffs"
current.loc[current["startTimeTBD"].astype(bool), "startTime"] = "TBD"

# ==================================================
# 🛠️ UTILITIES
# ==================================================

def get_team_color(team_name: str) -> str:
    """Return hex color for the team from the mapping table (fallback to gray)."""
    try:
        if team_name in map["cfb_name"].values:
            return map.loc[map["cfb_name"] == team_name, "Color1"].values[0]
    except Exception:
        pass
    return DEFAULT_COLOR


def bowl_probability_donut(probability: float, team_color: str) -> go.Figure:
    """Probability is expected in 0–1 scale."""
    percent = round(probability * 100, 1)
    fig = go.Figure(
        data=[
            go.Pie(
                values=[probability, 1 - probability],
                hole=0.5,
                marker_colors=[team_color, "#eee"],
                sort=False,
                direction="clockwise",
                textinfo="none",
            )
        ]
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(t=0, b=0, l=0, r=0),
        width=200,
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[
            dict(
                text=f"<b>{percent:.1f}%</b>",
                x=0.5,
                y=0.5,
                font_size=20,
                showarrow=False,
            )
        ],
    )
    return fig


def playoff_probability_donut(playoff_pct: float, team_color: str) -> go.Figure:
    """`playoff_pct` is expected in 0–100 scale (matches original simulate_full_season output)."""
    remaining = round(100 - playoff_pct, 1)
    fig = go.Figure(
        data=[
            go.Pie(
                labels=[f"Make Playoffs ({playoff_pct}%)", f"Miss Playoffs ({remaining}%)"],
                values=[playoff_pct, remaining],
                hole=0.5,
                marker_colors=[team_color, "#eee"],
                textinfo="label",
            )
        ]
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(t=0, b=0, l=0, r=0),
        width=200,
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[
            dict(
                text=f"<b>{playoff_pct:.1f}%</b>",
                x=0.5,
                y=0.5,
                font_size=20,
                showarrow=False,
            )
        ],
    )
    return fig


def simulate_single_team(current_df: pd.DataFrame, team: str, color: str) -> go.Figure:
    """Re-uses original simulation/plot logic, packaged into a function."""
    team_df = current_df[current_df["team"] == team].copy()
    team_df = team_df.sort_values("startDate")
    team_df["startDate"] = pd.to_datetime(team_df["startDate"]) 

    simulations = []
    num_simulations = 500

    for _ in range(num_simulations):
        df = team_df.copy()
        # Earned from completed wins
        df["actualEarned"] = df.apply(
            lambda row: row["actualPlayoffPoints"] if row["completed"] and row["result"] == "Win" else 0,
            axis=1,
        )

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
        fig.add_trace(
            go.Scatter(
                x=sim_dates,
                y=sim_points,
                mode="lines",
                line=dict(color=color, width=1),
                opacity=0.2,
                showlegend=False,
            )
        )

    # Reference lines
    fig.add_shape(
        type="line",
        x0=team_df["startDate"].min() - pd.Timedelta(days=6),
        x1=team_df["startDate"].max(),
        y0=120,
        y1=120,
        line=dict(color="red", width=2),
        layer="below",
    )
    fig.add_annotation(
        x=team_df["startDate"].min(),
        y=118,
        text="Playoff Conversation",
        showarrow=False,
        font=dict(color="red", size=12),
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.8)",
    )
    fig.add_shape(
        type="line",
        x0=team_df["startDate"].min() - pd.Timedelta(days=6),
        x1=team_df["startDate"].max(),
        y0=130,
        y1=130,
        line=dict(color="darkgreen", width=2),
        layer="below",
    )
    fig.add_annotation(
        x=team_df["startDate"].min(),
        y=132,
        text="Playoff Lock",
        showarrow=False,
        font=dict(color="darkgreen", size=12),
        xanchor="left",
        yanchor="bottom",
        bgcolor="rgba(255,255,255,0.8)",
    )

    fig.update_layout(
        title=f"{team} Simulated Seasons (500 Runs)",
        xaxis_title="Week",
        yaxis_title="Playoff Points",
        height=600,
    )
    return fig


def compute_bowl_odds(team_df: pd.DataFrame) -> float:
    """Return probability (0–1) of reaching 6 total wins based on remaining winProbs."""
    wins_so_far = (team_df["result"] == "Win").sum()
    upcoming = team_df[~team_df["completed"]].copy()
    upcoming["winProb"] = upcoming["winProb"].fillna(0.5)
    win_probs = pd.to_numeric(upcoming["winProb"], errors="coerce").dropna()

    if win_probs.empty:
        return 0.0

    n_sim = 10000
    sims = np.random.binomial(1, win_probs.values[:, None], size=(len(win_probs), n_sim))
    sim_wins = sims.sum(axis=0) + wins_so_far
    return float((sim_wins >= 6).mean())



def build_points_meter(team_df: pd.DataFrame, team_color: str) -> go.Figure:
    """Horizontal bar that fills to 130 with dark (earned) + light (projected) and a red 120 marker."""
    df = team_df.copy()
    df["startDate"] = pd.to_datetime(df["startDate"]) 

    # Earned so far (only completed wins)
    df["actualEarned"] = df.apply(
        lambda row: row.get("actualPlayoffPoints", 0) if row.get("completed") and row.get("result") == "Win" else 0,
        axis=1,
    )

    # Projected points (same approach you used before: only if winProb > 0.5)
    df["projectedEarned"] = df.apply(
        lambda row: (
            row.get("potentialPlayoffPoints", 0)
            + ((row.get("predictedPoints", 0) - row.get("predictedOpponentPoints", 0)) * COEF_MOV)
        )
        if (not row.get("completed")) and (row.get("winProb", 0.5) > 0.5)
        else 0,
        axis=1,
    )

    earned = float(df["actualEarned"].sum())
    projected = float(df["projectedEarned"].sum())

    # Cap everything at 130 max
    cap = 130.0
    actual_capped = min(earned, cap)
    total_after_projection = min(cap, earned + projected)
    projected_capped = max(0.0, total_after_projection - actual_capped)

    # Build a stacked horizontal bar
    fig = go.Figure()

    # Dark (earned)
    fig.add_trace(
        go.Bar(
            x=[actual_capped],
            y=[""],
            orientation="h",
            name="Earned",
            marker=dict(color=team_color),
            hovertemplate="Earned: %{x:.2f}<extra></extra>",
        )
    )

    # Light (projected)
    fig.add_trace(
        go.Bar(
            x=[projected_capped],
            y=[""],
            orientation="h",
            name="Projected",
            marker=dict(color=team_color, opacity=0.35),
            hovertemplate="Projected: %{x:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        barmode="stack",
        height=140,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
        xaxis=dict(range=[0, cap], title="Playoff Points"),
        yaxis=dict(showticklabels=False, showgrid=False),
        title="CFP Meter (Earned + Projected)",
    )

    # Red line at 120 (Playoff Conversation)
    fig.add_shape(
        type="line",
        x0=120, x1=120, y0=-0.5, y1=0.5,
        line=dict(color="red", width=2)
    )
    fig.add_annotation(
        x=120, y=0.6, text="Playoff Conversation",
        showarrow=False, font=dict(color="red", size=12),
        xanchor="center"
    )

    return fig


def prepare_current_tables(team_df: pd.DataFrame) -> pd.DataFrame:
    """Return the nicely formatted dataframe for the current season games section."""
    df = team_df.copy()
    df = df.sort_values("startDate")
    df["Score"] = df.apply(
        lambda row: "TBD"
        if pd.isna(row.get("teamPoints")) or pd.isna(row.get("opponentPoints"))
        else f"{int(row['teamPoints'])}–{int(row['opponentPoints'])}",
        axis=1,
    )
    df["Playoff Points"] = df.apply(
        lambda row: row.get("actualPlayoffPoints", 0) if row["completed"] else row.get("potentialPlayoffPoints", 0),
        axis=1,
    )
    df["Date"] = pd.to_datetime(df["startDate"]).dt.strftime("%B %d, %Y")
    df["Score Prediction"] = df.apply(
        lambda row: "N/A"
        if pd.isna(row.get("predictedPoints")) or pd.isna(row.get("predictedOpponentPoints"))
        else f"{int(round(row['predictedPoints']))}–{int(round(row['predictedOpponentPoints']))}",
        axis=1,
    )

    display_cols = [
        "Date",
        "startTime",
        "opponent",
        "location",
        "result",
        "Score",
        "winProb",
        "Score Prediction",
        "Playoff Points",
    ]

    # Style and rename for Streamlit display
    styled = (
        df[display_cols]
        .rename(
            columns={
                "winProb": "Win Prob",
                "startTime": "Time",
                "opponent": "Opponent",
                "location": "Location",
                "result": "Result",
            }
        )
        .style.format({"Win Prob": "{:.2%}", "Playoff Points": "{:.2f}"})
        .background_gradient(subset=["Win Prob"], cmap="RdYlGn", vmin=0.0, vmax=1.0)
    )
    return styled


def prepare_last_season_table(all_games: pd.DataFrame, team: str, season: int) -> pd.DataFrame:
    df = (
        all_games[
            (all_games["team"] == team)
            & (all_games["season"] == season)
            & (all_games["completed"])
        ]
        .sort_values("startDate", ascending=False)
        .copy()
    )
    if df.empty:
        return df

    df["Score"] = df.apply(
        lambda row: "TBD"
        if pd.isna(row.get("teamPoints")) or pd.isna(row.get("opponentPoints"))
        else f"{int(row['teamPoints'])}–{int(row['opponentPoints'])}",
        axis=1,
    )
    return df[["startDate", "startTime", "week", "opponent", "location", "result", "Score"]]


# ==================================================
# 🧭 LAYOUT: TEAM PICKER
# ==================================================
teams = sorted(current["team"].unique())
col_left, col_mid, col_right = st.columns([1, 2, 1])
with col_mid:
    selected_team = st.selectbox("Select a Team", ["-- Select a team --"] + teams, key="team_selector")

# ==================================================
# 📈 MAIN CONTENT WHEN TEAM IS SELECTED
# ==================================================
if selected_team and selected_team != "-- Select a team --":
    team_color = get_team_color(selected_team)

    # Slice once and reuse
    team_current = current[current["team"] == selected_team].copy()
    team_current = team_current.sort_values("startDate")
    team_current["startDate"] = pd.to_datetime(team_current["startDate"]) 

    # --- Bowl Odds (left) ---
    bowl_odds = compute_bowl_odds(team_current)
    with col_left:
        st.markdown("<div class='center' style='font-size:20px;'>🎯 <b>Bowl Eligibility Odds</b></div>", unsafe_allow_html=True)
        st.plotly_chart(bowl_probability_donut(bowl_odds, team_color), use_container_width=True)

    # --- Playoff Odds (right) ---
    # Use the authoritative season simulation output (same as original page)
    full_sim_df = simulate_full_season(current, win_model, win_model_features, num_simulations=200)
    team_row = full_sim_df[full_sim_df["team"] == selected_team]
    playoff_pct = float(team_row["playoff_pct"].values[0]) if not team_row.empty else 0.0

    with col_right:
        st.markdown("<div class='center' style='font-size:20px;'>🏆 <b>CFP Odds</b></div>", unsafe_allow_html=True)
        st.plotly_chart(playoff_probability_donut(playoff_pct, team_color), use_container_width=True)
        st.markdown(
            "<div class='center muted'>*Only 200 simulations for efficiency. See Season Simulation Tab for more Accurate Numbers</div>",
            unsafe_allow_html=True,
        )

    # --- Middle: Current vs Projected chart (cleaned) ---
    with col_mid:
        progress_fig = build_points_meter(team_current, team_color)
        st.plotly_chart(progress_fig, use_container_width=True)

    # ==================================================
    # 📋 Game Tables & Summary
    # ==================================================
    left_col, right_col = st.columns([1.2, 1])

    # 1) Current Season Table (left pane)
    with left_col:
        st.subheader(f"{team_current['season'].max()} Season: All Games for {selected_team}")

        if not team_current.empty:
            styled_df = prepare_current_tables(team_current)
            st.dataframe(styled_df, use_container_width=False, width=None, hide_index=True, height=460)
        else:
            st.info(f"No games found for {team_current['season'].max()}.")

        # 2) Last Season Completed Games
        st.subheader(f"🕰️ {team_current['season'].max() - 1} Season: Completed Games for {selected_team}")
        last_season_df = prepare_last_season_table(game_data, selected_team, team_current['season'].max() - 1)
        if not last_season_df.empty:
            st.dataframe(last_season_df, use_container_width=True, hide_index=True)
        else:
            st.info(f"No completed games found for {team_current['season'].max() - 1}.")

    # 3) Right pane: Metrics, Win Dist, and Single-Team Sims
    with right_col:
        st.subheader(f"🏅 {selected_team} Season Summary")

        # Record
        wins = int((team_current["result"] == "Win").sum())
        losses = int((team_current["result"] == "Loss").sum())

        # Rank (best next-week rank after last completed game)
        latest_rank = pd.Series(dtype=float)
        completed_weeks = team_current[team_current["completed"]]["week"]
        if not completed_weeks.empty:
            last_game_week = completed_weeks.max()
            # Next week's rank entry
            next_week_rows = team_current[team_current["week"] == last_game_week + 1].sort_values("startDate")
            latest_rank = next_week_rows["teamRank"].dropna().head(1)

        # Conference & Playoff odds from season sim
        conf_prob = float(team_row["conf_champ_pct"].values[0]) if not team_row.empty else 0.0

        m1, m2, m3, m4 = st.columns([1, 1, 2, 2])
        m1.metric(label="Record", value=f"{wins}-{losses}")
        m2.metric(label="Rank", value=f"#{int(latest_rank.values[0])}" if not latest_rank.empty else "N/A")
        m3.metric(label="Conf Champion %", value=f"{conf_prob:.2f}%")
        m4.metric(label="Playoff %", value=f"{playoff_pct:.2f}%")

        # Win total distribution for remaining games
        st.subheader("📈 Simulated Win Total Distribution")
        upcoming = team_current[~team_current["completed"]].copy()
        if not upcoming.empty:
            upcoming["winProb"] = upcoming["winProb"].fillna(0.5)
            win_probs = pd.to_numeric(upcoming["winProb"], errors="coerce").dropna()
            if not win_probs.empty:
                n_sim = 10000
                sims = np.random.binomial(1, win_probs.values[:, None], size=(len(win_probs), n_sim))
                sim_wins = sims.sum(axis=0) + wins
                win_dist = pd.Series(sim_wins).value_counts(normalize=True).sort_index()
                st.bar_chart((win_dist * 100).round(2), color=get_team_color(selected_team))
            else:
                st.info("No win probabilities available to simulate season.")
        else:
            st.info("No remaining games to simulate.")

        # Single-team Monte Carlo viz
        st.subheader("🎯 Single Team Season Simulations")
        st.plotly_chart(simulate_single_team(current, selected_team, team_color), use_container_width=True)
import sys
import os
sys.path.append(os.getcwd())
from functions import load_data

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

# ------------------------------
# PAGE CONFIGURATION & STYLING
# ------------------------------
st.set_page_config(page_title="College Football Playoff Dashboard", layout="wide")

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
        <h1 style="color:white; text-align:center;">🏈 College Football Playoff Dashboard</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# FUNCTION DEFINITIONS
# ------------------------------

# Load data
game_data, season_data, current, map = load_data()
logo_map = map[["cfb_name", "Logo", "Color1"]].rename(columns={"cfb_name": "Team"})
color_map = dict(zip(map["cfb_name"], map["Color1"].fillna("#808080")))

def get_latest_rank(df, rank_candidates=("teamRank","cfpRank","apRank","rank")):
    # choose the first rank column that exists
    rank_col = next((c for c in rank_candidates if c in df.columns), None)
    if rank_col is None:
        return pd.DataFrame(columns=["Team","Current Rank"])  # nothing to do

    tmp = df.loc[df[rank_col].notna(), ["team", "startDate", rank_col]].copy()
    if tmp.empty:
        return pd.DataFrame(columns=["Team","Current Rank"])

    tmp["startDate"] = pd.to_datetime(tmp["startDate"])
    tmp = tmp.sort_values(["team","startDate"])
    # take the most recent row per team where rank is present
    latest = tmp.groupby("team", as_index=False).tail(1)
    return latest.rename(columns={"team":"Team", rank_col:"Current Rank"})

# === FUNCTION IMPORTS ===
def get_playoff_points_tracker(game_data):
    teams = game_data["team"].unique()
    results = []

    for team in teams:
        team_games = game_data[game_data["team"] == team].copy()

        current_points = team_games[
            (team_games["completed"]) & (team_games["result"] == "Win")
        ]["actualPlayoffPoints"].sum()

        potential_points = team_games[
            (~team_games["completed"]) | (team_games["result"] == "Loss")
        ]["potentialPlayoffPoints"].sum()

        # Projected Points (original version)
        projected_points = current_points
        incomplete = team_games[~team_games["completed"]].copy()
        if not incomplete.empty:
            incomplete["winProb"] = incomplete["winProb"].fillna(0.5)
            incomplete["predictedMOV"] = (
                incomplete["predictedPoints"] - incomplete["predictedOpponentPoints"]
            ).fillna(0)

            COEF_MOV = 0.08237826714816475
            incomplete["predictedPlayoffPoints"] = (
                incomplete["potentialPlayoffPoints"] + (incomplete["predictedMOV"] * COEF_MOV)
            )
            projected_points += (incomplete["winProb"] * incomplete["predictedPlayoffPoints"]).sum()

        # 🔁 Projected Points 2 (only if winProb > 0.5)
        team_games["actualPlayoffPoints"] = team_games["actualPlayoffPoints"].fillna(0)
        team_games["potentialPlayoffPoints"] = team_games["potentialPlayoffPoints"].fillna(0)
        team_games["predictedMOV"] = (team_games["predictedPoints"] - team_games["predictedOpponentPoints"]).fillna(0)
        team_games["winProb"] = team_games["winProb"].fillna(0.5)

        team_games["actualEarned"] = team_games.apply(
            lambda row: row["actualPlayoffPoints"] if row["completed"] and row["result"] == "Win" else 0,
            axis=1
        )
        team_games["projEarned"] = team_games.apply(
            lambda row: (row["potentialPlayoffPoints"] + row["predictedMOV"] * COEF_MOV)
            if not row["completed"] and row["winProb"] > 0.5 else 0,
            axis=1
        )

        projected2 = (team_games["actualEarned"] + team_games["projEarned"]).sum()

        teamConference = team_games["teamConference"].mode().iloc[0] if not team_games["teamConference"].isna().all() else "Unknown"

        results.append({
            "Team": team,
            "Conference": teamConference,
            "Current Points": round(current_points, 2),
            "Base Potential Points": round(potential_points, 2),
            "Projected Points": round(projected_points, 2),
            "Projected Playoff Points 2": round(projected2, 2)  # 🔁 new column
        })

    df = pd.DataFrame(results)
    return df.sort_values("Projected Playoff Points 2", ascending=False)
def simulate_win_totals(df, num_simulations=5000):
    teams = df['team'].unique()
    results = []

    for team in teams:
        team_games = df[(df['team'] == team) & (~df['completed'])].copy()
        completed_wins = df[(df['team'] == team) & (df['completed']) & (df['result'] == "Win")].shape[0]

        if team_games.empty:
            total_wins = np.array([completed_wins] * num_simulations)
        else:
            win_probs = team_games["winProb"].fillna(0.5).values.reshape(-1, 1)
            sim = np.random.binomial(1, win_probs, (len(win_probs), num_simulations))
            total_wins = completed_wins + sim.sum(axis=0)

        results.append({
            "team": team,
            "win_10+": np.mean(total_wins >= 10),
            "win_11+": np.mean(total_wins >= 11),
            "win_12": np.mean(total_wins == 12)
        })

    return pd.DataFrame(results)
def build_line_data_from_current(current_df, selected_teams, color_dict, use_projection=True):
    current_df = current_df.copy()
    current_df["startDate"] = pd.to_datetime(current_df["startDate"])
    current_df = current_df[current_df["team"].isin(selected_teams)]
    current_df = current_df.sort_values(["team", "startDate"])
    current_df = current_df[current_df["seasonType"] == "regular"]

    COEF_MOV = 0.08237826714816475

    fig = go.Figure()

    final_points = []

    for team in selected_teams:
        team_df = current_df[current_df["team"] == team].copy()
        team_df = team_df.sort_values("startDate")

        team_df["actualPlayoffPoints"] = team_df["actualPlayoffPoints"].fillna(0)
        team_df["potentialPlayoffPoints"] = team_df["potentialPlayoffPoints"].fillna(0)
        if use_projection:
            team_df["predictedMOV"] = (team_df["predictedPoints"] - team_df["predictedOpponentPoints"]).fillna(0)
            team_df["winProb"] = team_df["winProb"].fillna(0.5)
        else:
            team_df["predictedMOV"] = 0
            team_df["winProb"] = 0.5

        team_df["actualEarned"] = team_df.apply(
            lambda row: row["actualPlayoffPoints"] if row["completed"] and row["result"] == "Win" else 0, axis=1
        )
        if use_projection:
            team_df["projEarned"] = team_df.apply(
                lambda row: (row["potentialPlayoffPoints"] + row["predictedMOV"] * COEF_MOV)
                if not row["completed"] and row["winProb"] > 0.5 else 0,
                axis=1
            )
        else:
            team_df["projEarned"] = 0
        team_df["cumulativePoints"] = (team_df["actualEarned"] + team_df["projEarned"]).cumsum()

        if not team_df.empty:
            start_anchor_time = team_df["startDate"].iloc[0] - pd.Timedelta(weeks=1)
            team_df = pd.concat([
                pd.DataFrame({"startDate":[start_anchor_time], "completed":[False],
                            "actualEarned":[0.0], "projEarned":[0.0], "cumulativePoints":[0.0]}),
                team_df
            ], ignore_index=True).sort_values("startDate")

        # --- Split traces robustly so completed is SOLID and projection is DASHED ---
        color = color_dict.get(team, "#808080")
        comp_mask = team_df["completed"].astype(bool).values if "completed" in team_df.columns else np.zeros(len(team_df), dtype=bool)

        if comp_mask.any():
            last_comp_idx = np.where(comp_mask)[0].max()
            # Solid: include anchor (row 0) through last completed row
            solid_df = team_df.iloc[: last_comp_idx + 1].copy()
            # Dashed: start at the LAST completed point (to connect), then only future rows
            dashed_df = team_df.iloc[last_comp_idx :].copy()
            # Remove any rows that are completed from the dashed segment EXCEPT the first row
            if len(dashed_df) > 1:
                dashed_df = pd.concat([dashed_df.iloc[[0]], dashed_df.iloc[1:][~dashed_df.iloc[1:]["completed"].astype(bool)]])
        else:
            # No completed games: solid is empty; dashed is entire series (anchor + future)
            solid_df = team_df.iloc[0:0].copy()
            dashed_df = team_df.copy()

        # Ensure strictly increasing time and no duplicates at the join
        solid_df = solid_df.sort_values("startDate")
        dashed_df = dashed_df.sort_values("startDate")
        if (not solid_df.empty) and (not dashed_df.empty):
            # Guarantee the first dashed point equals the last solid point
            dashed_df.iloc[0, dashed_df.columns.get_loc("startDate")] = solid_df["startDate"].iloc[-1]
            dashed_df.iloc[0, dashed_df.columns.get_loc("cumulativePoints")] = solid_df["cumulativePoints"].iloc[-1]

        # --- Draw SOLID completed line ---
        if not solid_df.empty:
            fig.add_trace(go.Scatter(
                x=solid_df["startDate"],
                y=solid_df["cumulativePoints"],
                mode="lines+markers",
                name=team,
                line=dict(color=color, width=2, dash="solid"),
                marker=dict(size=5),
                showlegend=False
            ))

        # --- Draw DASHED projection continuing from solid ---
        if (not dashed_df.empty) and (len(dashed_df) > 1 or solid_df.empty is True):
            fig.add_trace(go.Scatter(
                x=dashed_df["startDate"],
                y=dashed_df["cumulativePoints"],
                mode="lines+markers",
                name=team + " (Projected)",
                line=dict(color=color, width=2, dash="dash"),
                marker=dict(size=5),
                showlegend=False
            ))

        # Final label anchor
        if not dashed_df.empty:
            final_x = dashed_df["startDate"].iloc[-1]
            final_y = dashed_df["cumulativePoints"].iloc[-1]
        elif not solid_df.empty:
            final_x = solid_df["startDate"].iloc[-1]
            final_y = solid_df["cumulativePoints"].iloc[-1]
        else:
            continue

        final_points.append({
            "team": team,
            "x": final_x,
            "y": final_y,
            "color": color
        })
    # Sort bottom-up by y (lowest team first)
    final_points = sorted(final_points, key=lambda x: x["y"])

    y_positions = []
    y_tolerance = 5
    max_offset = 10

    for i, point in enumerate(final_points):
        desired_y = point["y"]
        candidate_y = desired_y

        # Try nudging up/down without breaking vertical order
        for offset in range(max_offset + 1):
            for direction in [1, -1]:  # Try up, then down
                trial_y = desired_y + direction * offset
                if all(abs(trial_y - y) >= y_tolerance for y in y_positions):
                    if i == 0 or trial_y > y_positions[-1]:  # Maintain increasing order
                        candidate_y = trial_y
                        break
            else:
                continue  # only executed if inner loop did not break
            break  # break outer loop if a good y was found

        y_positions.append(candidate_y)

        # Add team name text
        fig.add_annotation(
            x=point["x"],
            y=candidate_y,
            text=point["team"],
            showarrow=False,
            font=dict(color=point["color"], size=12),
            xanchor="left",
            yanchor="middle"
        )

    # Add red line for "Playoff Conversation"
    x_min = team_df["startDate"].min() - timedelta(days=21)
    x_max = team_df["startDate"].max() + timedelta(days=21)

    fig.add_shape(
        type="line",
        x0=x_min,
        x1=x_max,
        y0=120,
        y1=120,
        line=dict(color="red", width=2),
        layer="below"
    )
    fig.add_annotation(
        x=x_min,
        y=118,
        text="Playoff Conversation",
        showarrow=False,
        font=dict(color="red", size=12),
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.8)"
    )

    # Add dark green line for "Playoff Lock"
    fig.add_shape(
        type="line",
        x0=x_min,
        x1=x_max,
        y0=130,
        y1=130,
        line=dict(color="darkgreen", width=2),
        layer="below"
    )
    fig.add_annotation(
        x=x_min,
        y=132,
        text="Playoff Lock",
        showarrow=False,
        font=dict(color="darkgreen", size=12),
        xanchor="left",
        yanchor="bottom",
        bgcolor="rgba(255,255,255,0.8)"
    )

    fig.update_layout(
        title="Cumulative Playoff Points by Week",
        xaxis_title="Week",
        yaxis_title="Playoff Points",
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="left",
            x=0
        )
    )

    return fig

# ------------------------------
# PAGE LAYOUT STARTS HERE
# ------------------------------

# === TRACKER TABLE ===
st.subheader("📊 Playoff Points Tracker")
tracker_df = get_playoff_points_tracker(current)
sim_wins = simulate_win_totals(current)
tracker_df = tracker_df.merge(sim_wins, left_on="Team", right_on="team", how="left").drop(columns=["team"])
tracker_df = tracker_df.merge(logo_map, on="Team", how="left")
latest_rank = get_latest_rank(current)
tracker_df = tracker_df.merge(latest_rank, on="Team", how="left")

# Filters and Setup
column_renames = {
    "win_10+": "10+ Wins",
    "win_11+": "11+ Wins",
    "win_12": "12 Wins"
}
display_df = tracker_df.rename(columns=column_renames)
display_df["Current Rank"] = display_df["Current Rank"].fillna(len(display_df))
conferences = st.multiselect("Filter by Conference", display_df["Conference"].unique())
if conferences:
    display_df = display_df[display_df["Conference"].isin(conferences)]

columns_to_show = ["Logo", "Team", "Conference", "Current Points", "Base Potential Points", "Projected Points", "Projected Playoff Points 2", "10+ Wins", "11+ Wins", "12 Wins", "Current Rank"]
numeric_cols = [col for col in columns_to_show if col not in ["Logo", "Team", "Conference"]]
rank_column = st.selectbox("Rank by column:", numeric_cols)
display_df["Rank"] = display_df[rank_column].rank(ascending=False, method="min").astype(int)
columns_to_show = ["Rank"] + columns_to_show

format_dict = {
    "Current Points": "{:.2f}",
    "Base Potential Points": "{:.2f}",
    "Projected Points": "{:.2f}",
    "Projected Playoff Points 2": "{:.2f}",
    "10+ Wins": "{:.1%}",
    "11+ Wins": "{:.1%}",
    "12 Wins": "{:.1%}",
    "Current Rank": "{:.0f}"
}

highlight_cols = [col for col in columns_to_show if col in format_dict]
normal_cols = [c for c in highlight_cols if c != "Current Rank"]
styled = display_df[columns_to_show].style.format(format_dict).background_gradient(cmap="RdYlGn", subset=normal_cols)
styled = styled.background_gradient(cmap="RdYlGn_r", subset=["Current Rank"])

st.dataframe(styled, column_config={
    "Logo": st.column_config.ImageColumn("Logo", help="Team Logos")
}, use_container_width=False, height=600, hide_index=True)

# === CHARTS ===
st.markdown("### 📈 Cumulative Playoff Points by Week")
available_years = sorted(game_data["season"].dropna().unique().tolist(), reverse=True)
view_year = st.selectbox("Select a season to view (optional):", ["Current Season"] + [str(y) for y in available_years])
all_conferences = sorted(tracker_df["Conference"].dropna().unique().tolist())
default_conf = st.selectbox("Quick select by conference:", ["(None)"] + all_conferences)

if default_conf != "(None)":
    top_teams = tracker_df[tracker_df["Conference"] == default_conf]["Team"].sort_values().tolist()
else:
    top_teams = tracker_df.sort_values("Projected Points", ascending=False)["Team"].head(5).tolist()

selected_teams = st.multiselect("Select teams to show:", tracker_df["Team"].sort_values().tolist(), default=top_teams)
data_source = current if view_year == "Current Season" else game_data[game_data["season"] == int(view_year)].copy()
use_projection = view_year == "Current Season"
color_dict = dict(zip(logo_map["Team"], logo_map["Color1"]))

line_chart = build_line_data_from_current(data_source, selected_teams, color_dict, use_projection)
st.plotly_chart(line_chart, use_container_width=True)
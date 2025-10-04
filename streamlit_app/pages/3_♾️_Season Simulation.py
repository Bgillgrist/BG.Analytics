# =========================
# 📦 IMPORTS & CONFIG
# =========================
import sys
import os
sys.path.append(os.getcwd())
from functions import load_data, train_win_model, apply_win_model, train_total_model, train_spread_model, simulate_full_season

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Full Season Simulation", layout="wide")
st.title("🧮 Full Season Simulation")

# 🔁 Re-run control (clear caches, then rerun)
if st.button("🔁 Re-run simulation", help="Run the Monte Carlo again with fresh randomness"):
    # Clear any cached @st.cache_data / @st.cache_resource results
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# =========================
# 📥 LOAD DATA
# =========================
game_data, season_data, current, map = load_data()

# =========================
# ⚙️ SIMULATION FUNCTION
# =========================
win_model, win_model_features = train_win_model(game_data)
#model_team, expected_features = train_total_model(game_data)
#model_team, expected_features = train_spread_model(game_data)

full_sim_df = simulate_full_season(current, win_model, win_model_features)

# =========================
# 🎛️ FILTERS
# =========================
conferences = ["All"] + sorted(full_sim_df["conference"].unique())
selected_conf = st.selectbox("📍 Filter by Conference", conferences)

sim_df = full_sim_df.copy()
if selected_conf != "All":
    sim_df = sim_df[sim_df["conference"] == selected_conf]

# =========================
# 📊 DISPLAY STANDINGS
# =========================

col_left, col_right = st.columns([1, 1])

# Left Side Chart Standings

with col_left:

    st.subheader(f"🥇🥈🥉 {selected_conf} Projected Standings")

    sim_df = sim_df.merge(map, left_on="team", right_on="cfb_name", how="left")

    # Sort by average wins
    sim_df = sim_df.sort_values("wins_avg", ascending=False)
    top_20 = sim_df.sort_values("wins_avg", ascending=False).head(20)

    import plotly.graph_objects as go

    fig = go.Figure()

    # --- 95% band (middle) ---
    fig.add_trace(go.Bar(
        x=(top_20["wins_ceil_95"] - top_20["wins_floor_95"]).astype(float),
        y=top_20["team"],
        base=top_20["wins_floor_95"].astype(float),
        orientation='h',
        marker_color= top_20["Color1"],
        opacity= .5,
        name='95% Confidence',
        hovertemplate='95% band: %{base:.1f}–%{customdata:.1f}<extra></extra>',
        customdata=top_20["wins_ceil_95"].astype(float),
        showlegend=True
    ))

    # --- 75% band (inside) ---
    fig.add_trace(go.Bar(
        x=(top_20["wins_ceil_75"] - top_20["wins_floor_75"]).astype(float),
        y=top_20["team"],
        base=top_20["wins_floor_75"].astype(float),
        orientation='h',
        marker_color= top_20["Color1"],
        opacity= .9,
        name='75% Confidence',
        hovertemplate='75% band: %{base:.1f}–%{customdata:.1f}<extra></extra>',
        customdata=top_20["wins_ceil_75"].astype(float),
        showlegend=True
    ))

    # Optional: a small black dot at Avgerage Wins
    fig.add_trace(go.Scatter(
        x=top_20["wins_avg"],
        y=top_20["team"],
        mode="markers",
        marker=dict(size=6, color="black"),
        name="wins_avg",
        showlegend=False
    ))

    fig.update_layout(
        title="Projected Wins with Difference Confidence Levels",
        yaxis_title="team",
        xaxis_title="Projected Wins",
        height=600,
        margin=dict(t=90, l=150, r=30, b=50),
        barmode='overlay',                 # key so bands sit behind the avg bar
        xaxis=dict(
            tickmode="linear",
            tick0=0,
            dtick=1,
            showgrid=True,
            gridcolor="lightgrey",
            gridwidth=1,
            griddash="dash"
        ),
        yaxis=dict(
            autorange="reversed",
            range=[-0.5, 19.5],
            fixedrange=False
        ),
        legend=dict(
            orientation="h",        # horizontal legend
            yanchor="bottom",
            y=1.02,                 # place above plotting area
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="rgba(0,0,0,0)",
            borderwidth=0
        )
    )

    st.plotly_chart(fig, use_container_width=True)

# Right side table standings

with col_right:
    # Select and rename columns for display
    display_df = (
        sim_df[[
            "team", "Logo", "wins_avg", "losses_avg",
            "conf_wins_avg", "conf_champ_pct", "pp_avg", "playoff_pct", "auto_bid_pct", "at_large_pct"
        ]]
        .rename(columns={
            "team": "Team",
            "wins_avg": "Avg Wins",
            "losses_avg": "Avg Losses",
            "conf_wins_avg": "Conf Wins",
            "conf_champ_pct": "Conf Champ %",
            "pp_avg": "Playoff Points",
            "playoff_pct": "Playoff %",
            "auto_bid_pct": "Auto Bid %",
            "at_large_pct": "At Large Bid %",
        })
    )

    styler = (
        display_df.style
        .background_gradient(cmap="RdYlGn")  # applies to all numerics
        .background_gradient(cmap="RdYlGn_r", subset=["Avg Losses"])  # reversed just for losses
    )

    st.subheader(" ")
    st.dataframe(
        styler,
        column_config={
            "Logo": st.column_config.ImageColumn("Logo", help="Team Logos"),
            "Avg Wins": st.column_config.NumberColumn("Avg Wins", format="%.1f"),
            "Avg Losses": st.column_config.NumberColumn("Avg Losses", format="%.1f"),
            "Conf Wins": st.column_config.NumberColumn("Conf Wins", format="%.1f"),
            "Playoff Points": st.column_config.NumberColumn("Playoff Points", format="%.2f"),
            "Conf Champ %": st.column_config.NumberColumn("Conf Champ %", format="%.1f%%"),
            "Playoff %": st.column_config.NumberColumn("Playoff %", format="%.1f%%"),
            "Auto Bid %": st.column_config.NumberColumn("Auto Bid %", format="%.1f%%"),
            "At Large Bid %": st.column_config.NumberColumn("At Large Bid %", format="%.1f%%"),
        },
        height=600,
        use_container_width=True,
        hide_index=True
    )

# =========================
# 🏆 Conference Championship Projections
# =========================
st.markdown("""---""")

st.subheader("📋 Conference Championship Matchups & Predictions")

conference_rows = [
    ["SEC", "Big Ten", "Big 12"],
    ["ACC", "Mountain West", "American Athletic"],
    ["Sun Belt", "Conference USA", "Mid-American"]
]

def get_team_color(team, col="Color1"):
    row = map[map["cfb_name"] == team]
    if not row.empty:
        return row.iloc[0][col]
    return "#cccccc"

for conf_list in conference_rows:
    cols = st.columns(len(conf_list))  # 4 per row or however many in this row

    for i, conf in enumerate(conf_list):
        with cols[i]:
            # Filter for this conference
            conf_teams = full_sim_df[full_sim_df["conference"] == conf]
            if conf_teams.empty:
                st.markdown(f"### {conf}\n_No data available_")
                continue

            # Keep teams with nonzero probability
            conf_teams = conf_teams[conf_teams["conf_champ_pct"] > 0]

            if len(conf_teams) == 0:
                st.markdown(f"### {conf}\n_No projected champion_")
                continue

            # Separate major contenders vs others
            major = conf_teams[conf_teams["conf_champ_pct"] > 3].copy()
            other = conf_teams[conf_teams["conf_champ_pct"] <= 3].copy()

            # Prepare chart data
            labels = major["team"].tolist()
            values = major["conf_champ_pct"].tolist()
            colors = [get_team_color(team, "Color1") for team in labels]

            if not other.empty:
                labels.append("Other")
                values.append(round(other["conf_champ_pct"].sum(), 1))
                colors.append("#999999")  # Gray for Other

            # Build chart
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker=dict(
                    colors=colors,
                    line=dict(color="white", width=1)
                ),
                textinfo='label+percent'
            )])

            major = major.sort_values("conf_champ_pct", ascending=False)
            top_team = major.iloc[0]["team"] if not major.empty else conf_teams.sort_values("conf_champ_pct", ascending=False).iloc[0]["team"]
            logo_url = map[map["cfb_name"] == top_team]["Logo"].values[0] if top_team in map["cfb_name"].values else None
            if logo_url:
                fig.update_layout(images=[dict(
                    source=logo_url,
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    sizex=0.3, sizey=0.3,
                    xanchor="center", yanchor="middle",
                    layer="above"
                )])

            fig.update_layout(
                title=dict(
                    text=f"{conf} Champion Odds",
                    x=0.5,
                    xanchor='center',
                    font=dict(size=28)
                ),
                margin=dict(t=40, b=10, l=10, r=10),
                height=350,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

# =========================
# 🏆 CFP Predictions
# =========================
st.markdown("""---""")
st.subheader("🏆 College Football Playoff Predictions")

full_sim_df["Max Conf Champ %"] = full_sim_df.groupby("conference")["conf_champ_pct"].transform("max")
full_sim_df["Proj Champ"] = np.where(
    full_sim_df["conf_champ_pct"] == full_sim_df["Max Conf Champ %"],
    "Yes",
    "No"
)
full_sim_df.drop(columns="Max Conf Champ %", inplace=True)

# Select Playoff Teams
# For each conference, keep the team with highest auto-bid %
champs_pool = full_sim_df[(full_sim_df["Proj Champ"] == "Yes")]
best_champs = (
    champs_pool.sort_values("auto_bid_pct", ascending=False)
    .groupby("conference", as_index=False)
    .first()
)

# Then take the 5 highest pp_avg from those conference winners
top_5_champs = best_champs.sort_values("pp_avg", ascending=False).head(5)
remaining_df = full_sim_df[full_sim_df["Proj Champ"] == "No"]
at_large_df = remaining_df.sort_values("pp_avg", ascending=False).head(7)

final_12_df = pd.concat([top_5_champs, at_large_df]).sort_values("pp_avg", ascending=False).reset_index(drop=True)
final_12_df["Seed"] = list(range(1, 13))
final_12_df = final_12_df.sort_values("Seed")

first_round_matchups = [
    (5, 12), (6, 11), (7, 10), (8, 9)
]

latest_ratings = (
    current
    .sort_values("startDate", ascending=False)
    .dropna(subset=["teamPregameRating"])
    .drop_duplicates(subset="team", keep="first")[["team", "teamPregameRating"]]
    .rename(columns={"teamPregameRating": "PregameRating"})
)

rating_dict = dict(zip(latest_ratings["team"], latest_ratings["PregameRating"]))
final_12_df = final_12_df.merge(latest_ratings, on="team", how="left")
seed_lookup = dict(zip(final_12_df["team"], final_12_df["Seed"]))

round_1_games = []
for seed1, seed2 in first_round_matchups:
    team1 = final_12_df[final_12_df["Seed"] == seed1].iloc[0]["team"]
    team2 = final_12_df[final_12_df["Seed"] == seed2].iloc[0]["team"]

    round_1_games.append({
        "team": team1,
        "opponent": team2,
        "teamPregameRating": rating_dict.get(team1, np.nan),
        "opponentPregameRating": rating_dict.get(team2, np.nan),
        "location": "Neutral",
        "Seed 1": seed1,
        "Seed 2": seed2
    })

round_1_df = pd.DataFrame(round_1_games)

model, features = train_win_model(game_data)

round_1_probs = apply_win_model(round_1_df, model, features)
round_1_df["Win Prob"] = round_1_probs
round_1_df["Projected Winner"] = np.where(round_1_probs >= 0.5, round_1_df["team"], round_1_df["opponent"])

winners_dict = {}
winner_seeds = {}

for _, row in round_1_df.iterrows():
    winner = row["Projected Winner"]
    loser = row["team"] if winner == row["opponent"] else row["opponent"]

    # Pull both seeds
    seed_team = row["Seed 1"]
    seed_opp = row["Seed 2"]

    # Assign correct seed to winner
    winner_seed = seed_team if winner == row["team"] else seed_opp

    matchup = tuple(sorted([seed_team, seed_opp]))
    winners_dict[matchup] = winner
    winner_seeds[winner] = winner_seed
    for i in range(1, 5):
        team = final_12_df[final_12_df["Seed"] == i].iloc[0]["team"]
        winner_seeds[team] = i

quarterfinal_matchups = [
    (1, winners_dict.get((8, 9))),
    (4, winners_dict.get((5, 12))),
    (3, winners_dict.get((6, 11))),
    (2, winners_dict.get((7, 10)))
]

qf_games = []
for top_seed, opponent_team in quarterfinal_matchups:
    top_team = final_12_df[final_12_df["Seed"] == top_seed].iloc[0]["team"]

    qf_games.append({
        "team": top_team,
        "opponent": opponent_team,
        "teamPregameRating": rating_dict.get(top_team, np.nan),
        "opponentPregameRating": rating_dict.get(opponent_team, np.nan),
        "location": "Neutral",
        "Seed 1": seed_lookup.get(top_team),
        "Seed 2": seed_lookup.get(opponent_team)
    })

qf_df = pd.DataFrame(qf_games)
qf_probs = apply_win_model(qf_df, model, features)
qf_df["Win Prob"] = qf_probs
qf_df["Projected Winner"] = np.where(qf_probs >= 0.5, qf_df["team"], qf_df["opponent"])
qf_df["winnerSeed"] = qf_df["Projected Winner"].map(winner_seeds)

######### Semifinals Matchups ##########

semifinal_games = []

# Map QF winners by order
qf_winners = list(qf_df["Projected Winner"])

# Matchups: (QF1 vs QF2), (QF3 vs QF4)
sf_matchups = [
    (qf_winners[0], qf_winners[1]),  # SF1
    (qf_winners[2], qf_winners[3])   # SF2
]

for team1, team2 in sf_matchups:
    semifinal_games.append({
        "team": team1,
        "opponent": team2,
        "teamPregameRating": rating_dict.get(team1, np.nan),
        "opponentPregameRating": rating_dict.get(team2, np.nan),
        "location": "Neutral",
        "Seed 1": winner_seeds.get(team1, np.nan),
        "Seed 2": winner_seeds.get(team2, np.nan)
    })

semifinal_df = pd.DataFrame(semifinal_games)

# Predict outcomes
semifinal_probs = apply_win_model(semifinal_df, model, features)
semifinal_df["Win Prob"] = semifinal_probs
semifinal_df["Projected Winner"] = np.where(semifinal_probs >= 0.5, semifinal_df["team"], semifinal_df["opponent"])

######### Finals Matchup ##########

final_game = []

# Map QF winners by order
sf_winners = list(semifinal_df["Projected Winner"])

# Matchup
final_matchup = [
    (sf_winners[0], sf_winners[1])
]

for team1, team2 in final_matchup:
    final_game.append({
        "team": team1,
        "opponent": team2,
        "teamPregameRating": rating_dict.get(team1, np.nan),
        "opponentPregameRating": rating_dict.get(team2, np.nan),
        "location": "Neutral",
        "Seed 1": winner_seeds.get(team1, np.nan),
        "Seed 2": winner_seeds.get(team2, np.nan)
    })

final_df = pd.DataFrame(final_game)

# Predict outcomes
final_prob = apply_win_model(final_df, model, features)
final_df["Win Prob"] = final_prob
final_df["Projected Winner"] = np.where(final_prob >= 0.5, final_df["team"], final_df["opponent"])

projected_winner = final_df["Projected Winner"][0]

# CREATE BRACKET

team_logos = dict(zip(map["cfb_name"], map["Logo"]))

col1, col2, col3, col4, col5, col6, col7 = st.columns([1,1,1,1,1,1,1])

with col1:
    seed_9 = final_12_df[final_12_df["Seed"] == 9]["team"].values[0]
    seed_8 = final_12_df[final_12_df["Seed"] == 8]["team"].values[0]
    
    matchup_df = pd.DataFrame([seed_9, seed_8])

    st.markdown(f"""
        <table style="width: 100%; border-collapse: collapse; font-size: 16px;">
            <tr style="height: 4em;">
            <td style="padding: 8px; text-align: center; vertical-align: middle;"><b>{9}</b></td>
            <td style="padding: 8px; text-align: left; vertical-align: middle;">{seed_9}</td>
            <td style="padding: 8px; text-align: center; vertical-align: middle;">
                <img src="{team_logos.get(seed_9, '')}" alt="{seed_9}" style="height: 30px;">
            </td>
        </tr>
            <tr style="height: 4em;">
            <td style="padding: 8px; text-align: center; vertical-align: middle;"><b>{8}</b></td>
            <td style="padding: 8px; text-align: left; vertical-align: middle;">{seed_8}</td>
            <td style="padding: 8px; text-align: center; vertical-align: middle;">
                <img src="{team_logos.get(seed_8, '')}" alt="{seed_8}" style="height: 30px;">
            </td>
        </tr>
        </table>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)

    seed_12 = final_12_df[final_12_df["Seed"] == 12]["team"].values[0]
    seed_5 = final_12_df[final_12_df["Seed"] == 5]["team"].values[0]
    
    matchup_df = pd.DataFrame([seed_12, seed_5])

    st.markdown(f"""
        <table style="width: 100%; border-collapse: collapse; font-size: 16px;">
            <tr style="height: 4em;">
            <td style="padding: 8px; text-align: center; vertical-align: middle;"><b>{12}</b></td>
            <td style="padding: 8px; text-align: left; vertical-align: middle;">{seed_12}</td>
            <td style="padding: 8px; text-align: center; vertical-align: middle;">
                <img src="{team_logos.get(seed_12, '')}" alt="{seed_12}" style="height: 30px;">
            </td>
        </tr>
            <tr style="height: 4em;">
            <td style="padding: 8px; text-align: center; vertical-align: middle;"><b>{5}</b></td>
            <td style="padding: 8px; text-align: left; vertical-align: middle;">{seed_5}</td>
            <td style="padding: 8px; text-align: center; vertical-align: middle;">
                <img src="{team_logos.get(seed_5, '')}" alt="{seed_5}" style="height: 30px;">
            </td>
        </tr>
        </table>
        """, unsafe_allow_html=True)
    

    next2 = full_sim_df[~full_sim_df["team"].isin(final_12_df["team"].tolist())].sort_values("pp_avg", ascending=False).head(2)
    st.markdown("### First 2 Out:")
    st.markdown(f"""
        <table style="width: 100%; border-collapse: collapse; font-size: 16px;">
            <tr style="height: 4em;">
            <td style="padding: 8px; text-align: center; vertical-align: middle;"><b>{13}</b></td>
            <td style="padding: 8px; text-align: left; vertical-align: middle;">{next2.iloc[0]["team"]}</td>
            <td style="padding: 8px; text-align: center; vertical-align: middle;">
                <img src="{team_logos.get(next2.iloc[0]["team"], '')}" alt="{next2.iloc[0]["team"]}" style="height: 30px;">
            </td>

        </tr>
            <tr style="height: 4em;">
            <td style="padding: 8px; text-align: center; vertical-align: middle;"><b>{14}</b></td>
            <td style="padding: 8px; text-align: left; vertical-align: middle;">{next2.iloc[1]["team"]}</td>
            <td style="padding: 8px; text-align: center; vertical-align: middle;">
                <img src="{team_logos.get(next2.iloc[1]["team"], '')}" alt="{next2.iloc[1]["team"]}" style="height: 30px;">
            </td>
        </tr>
        </table>
        """, unsafe_allow_html=True)
    
with col2:
    st.markdown("<br>", unsafe_allow_html=True)

    qf1_team1 = quarterfinal_matchups[0][0]
    qf1_team2 = quarterfinal_matchups[0][1]

    team1 = final_12_df[final_12_df["Seed"] == qf1_team1]["team"].values[0]
    team2 = quarterfinal_matchups[0][1]

    st.markdown(f"""
        <table style="width: 100%; border-collapse: collapse; font-size: 16px;">
        <tr style="height: 4em;">
            <td style="padding: 8px; text-align: center; vertical-align: middle;"><b>{1}</b></td>
            <td style="padding: 8px; vertical-align: middle;">{team1}</td>
            <td style="padding: 8px; text-align: center; vertical-align: middle;">
                <img src="{team_logos.get(team1, '')}" alt="{team1}" style="height: 30px;">
            </td>
        </tr>
        <tr style="height: 4em;">
            <td style="padding: 8px; text-align: center;"><b>{winner_seeds.get(team2, '?')}</b></td>
            <td style="padding: 8px; vertical-align: middle;">{team2}</td>
            <td style="padding: 8px; text-align: center; vertical-align: middle;">
                <img src="{team_logos.get(team2, '')}" alt="{team2}" style="height: 30px;">
            </td>
        </tr>
        </table>
    """, unsafe_allow_html=True)

    st.markdown("<br><br><br><br>", unsafe_allow_html=True)

    qf1_team3 = quarterfinal_matchups[1][0]
    qf1_team4 = quarterfinal_matchups[1][1]

    team3 = final_12_df[final_12_df["Seed"] == qf1_team3]["team"].values[0]
    team4 = quarterfinal_matchups[1][1]

    st.markdown(f"""
        <table style="width: 100%; border-collapse: collapse; font-size: 16px;">
        <tr style="height: 4em;">
            <td style="padding: 8px; text-align: center; vertical-align: middle;"><b>{4}</b></td>
            <td style="padding: 8px; vertical-align: middle;">{team3}</td>
            <td style="padding: 8px; text-align: center; vertical-align: middle;">
                <img src="{team_logos.get(team3, '')}" alt="{team3}" style="height: 30px;">
            </td>
        </tr>
        <tr style="height: 4em;">
            <td style="padding: 8px; text-align: center;"><b>{winner_seeds.get(team4, '?')}</b></td>
            <td style="padding: 8px; vertical-align: middle;">{team4}</td>
            <td style="padding: 8px; text-align: center; vertical-align: middle;">
                <img src="{team_logos.get(team4, '')}" alt="{team4}" style="height: 30px;">
            </td>
        </tr>
        </table>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)

    sf1_team1 = sf_matchups[0][0]
    sf1_team2 = sf_matchups[0][1]

    st.markdown(f"""
        <table style="width: 100%; border-collapse: collapse; font-size: 16px;">
        <tr style="height: 4em;">
            <td style="padding: 8px; text-align: center; vertical-align: middle;"><b>{winner_seeds.get(sf1_team1, '?')}</b></td>
            <td style="padding: 8px; vertical-align: middle;">{sf1_team1}</td>
            <td style="padding: 8px; text-align: center; vertical-align: middle;">
                <img src="{team_logos.get(sf1_team1, '')}" alt="{sf1_team1}" style="height: 30px;">
            </td>
        </tr>
        <tr style="height: 4em;">
            <td style="padding: 8px; text-align: center;"><b>{winner_seeds.get(sf1_team2, '?')}</b></td>
            <td style="padding: 8px; vertical-align: middle;">{sf1_team2}</td>
            <td style="padding: 8px; text-align: center; vertical-align: middle;">
                <img src="{team_logos.get(sf1_team2, '')}" alt="{sf1_team2}" style="height: 30px;">
            </td>
        </tr>
        </table>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)

    final_team1 = final_matchup[0][0]
    final_team2 = final_matchup[0][1]

    st.markdown(f"""
        <table style="width: 100%; border-collapse: collapse; font-size: 16px;">
        <tr style="height: 4em;">
            <td style="padding: 8px; text-align: center; vertical-align: middle;"><b>{winner_seeds.get(final_team1, '?')}</b></td>
            <td style="padding: 8px; vertical-align: middle;">{final_team1}</td>
            <td style="padding: 8px; text-align: center; vertical-align: middle;">
                <img src="{team_logos.get(final_team1, '')}" alt="{final_team1}" style="height: 30px;">
            </td>
        </tr>
        <tr style="height: 4em;">
            <td style="padding: 8px; text-align: center;"><b>{winner_seeds.get(final_team2, '?')}</b></td>
            <td style="padding: 8px; vertical-align: middle;">{final_team2}</td>
            <td style="padding: 8px; text-align: center; vertical-align: middle;">
                <img src="{team_logos.get(final_team2, '')}" alt="{final_team2}" style="height: 30px;">
            </td>
        </tr>
        </table>
    """, unsafe_allow_html=True)

    st.markdown(
        f"""
        <div style="text-align: center;">
            <span style="font-size: 22px;">Projected Winner:</span><br>
            <b style="font-size: 28px;">{projected_winner}</b><br>
            <img src="{team_logos.get(projected_winner, "")}" alt="{projected_winner}" style="height: 60px; margin-top: 10px;">
        </div>
        """,
        unsafe_allow_html=True
    )


with col5:
    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)

    sf2_team1 = sf_matchups[1][1]
    sf2_team2 = sf_matchups[1][0]

    st.markdown(f"""
        <table style="width: 100%; border-collapse: collapse; font-size: 16px;">
        <tr style="height: 4em;">
            <td style="padding: 8px; text-align: center; vertical-align: middle;">
                <img src="{team_logos.get(sf2_team1, '')}" alt="{sf2_team1}" style="height: 30px;">
            </td>
            <td style="padding: 8px; vertical-align: middle;">{sf2_team1}</td>
            <td style="padding: 8px; text-align: center; vertical-align: middle;"><b>{winner_seeds.get(sf2_team1, '?')}</b></td>
        </tr>
        <tr style="height: 4em;">
            <td style="padding: 8px; text-align: center; vertical-align: middle;">
                <img src="{team_logos.get(sf2_team2, '')}" alt="{sf2_team2}" style="height: 30px;">
            </td>
            <td style="padding: 8px; vertical-align: middle;">{sf2_team2}</td>
            <td style="padding: 8px; text-align: center; vertical-align: middle;"><b>{winner_seeds.get(sf2_team2, '?')}</b></td>
        </tr>
        </table>
    """, unsafe_allow_html=True)

with col6:
    st.markdown("<br>", unsafe_allow_html=True)

    qf1_team1 = quarterfinal_matchups[3][0]
    qf1_team2 = quarterfinal_matchups[3][1]

    team1 = final_12_df[final_12_df["Seed"] == qf1_team1]["team"].values[0]
    team2 = quarterfinal_matchups[3][1]

    st.markdown(f"""
        <table style="width: 100%; border-collapse: collapse; font-size: 16px;">
        <tr style="height: 4em;">
            <td style="padding: 8px; text-align: center; vertical-align: middle;">
                <img src="{team_logos.get(team1, '')}" alt="{team1}" style="height: 30px;">
            </td>
            <td style="padding: 8px; vertical-align: middle;">{team1}</td>
            <td style="padding: 8px; text-align: center; vertical-align: middle;"><b>{2}</b></td>
        </tr>
        <tr style="height: 4em;">
            <td style="padding: 8px; text-align: center; vertical-align: middle;">
                <img src="{team_logos.get(team2, '')}" alt="{team2}" style="height: 30px;">
            </td>
            <td style="padding: 8px; vertical-align: middle;">{team2}</td>
            <td style="padding: 8px; text-align: center;"><b>{winner_seeds.get(team2, '?')}</b></td>
        </tr>
        </table>
    """, unsafe_allow_html=True)

    st.markdown("<br><br><br><br>", unsafe_allow_html=True)

    qf1_team3 = quarterfinal_matchups[2][0]
    qf1_team4 = quarterfinal_matchups[2][1]

    team3 = final_12_df[final_12_df["Seed"] == qf1_team3]["team"].values[0]
    team4 = quarterfinal_matchups[2][1]

    st.markdown(f"""
        <table style="width: 100%; border-collapse: collapse; font-size: 16px;">
        <tr style="height: 4em;">
            <td style="padding: 8px; text-align: center; vertical-align: middle;">
                <img src="{team_logos.get(team3, '')}" alt="{team3}" style="height: 30px;">
            </td>
            <td style="padding: 8px; vertical-align: middle;">{team3}</td>
            <td style="padding: 8px; text-align: center; vertical-align: middle;"><b>{3}</b></td>
        </tr>
        <tr style="height: 4em;">
            <td style="padding: 8px; text-align: center; vertical-align: middle;">
                <img src="{team_logos.get(team4, '')}" alt="{team4}" style="height: 30px;">
            </td>
            <td style="padding: 8px; vertical-align: middle;">{team4}</td>
            <td style="padding: 8px; text-align: center;"><b>{winner_seeds.get(team4, '?')}</b></td>
        </tr>
        </table>
    """, unsafe_allow_html=True)
    
with col7:
    seed_10 = final_12_df[final_12_df["Seed"] == 10]["team"].values[0]
    seed_7 = final_12_df[final_12_df["Seed"] == 7]["team"].values[0]
    
    matchup_df = pd.DataFrame([seed_10, seed_7])

    st.markdown(f"""
        <table style="width: 100%; border-collapse: collapse; font-size: 16px;">
            <tr style="height: 4em;">
                <td style="padding: 8px; text-align: center; vertical-align: middle;">
                <img src="{team_logos.get(seed_10, '')}" alt="{seed_10}" style="height: 30px;">
            </td>
                <td style="padding: 8px; text-align: left; vertical-align: middle;">{seed_10}</td>
                <td style="padding: 8px; text-align: center; vertical-align: middle;"><b>{10}</b></td>
            </tr>
            <tr style="height: 4em;">
                <td style="padding: 8px; text-align: center; vertical-align: middle;">
                <img src="{team_logos.get(seed_7, '')}" alt="{seed_7}" style="height: 30px;">
            </td>
                <td style="padding: 8px; text-align: left; vertical-align: middle;">{seed_7}</td>
                <td style="padding: 8px; text-align: center; vertical-align: middle;"><b>{7}</b></td>
            </tr>
        </table>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)

    seed_11 = final_12_df[final_12_df["Seed"] == 11]["team"].values[0]
    seed_6 = final_12_df[final_12_df["Seed"] == 6]["team"].values[0]
    
    matchup_df = pd.DataFrame([seed_11, seed_6])

    st.markdown(f"""
        <table style="width: 100%; border-collapse: collapse; font-size: 16px;">
            <tr style="height: 4em;">
            <td style="padding: 8px; text-align: center; vertical-align: middle;">
                <img src="{team_logos.get(seed_11, '')}" alt="{seed_11}" style="height: 30px;">
            </td>
            <td style="padding: 8px; text-align: left; vertical-align: middle;">{seed_11}</td>
            <td style="padding: 8px; text-align: center; vertical-align: middle;"><b>{11}</b></td>
        </tr>
            <tr style="height: 4em;">
            <td style="padding: 8px; text-align: center; vertical-align: middle;">
                <img src="{team_logos.get(seed_6, '')}" alt="{seed_6}" style="height: 30px;">
            </td>
            <td style="padding: 8px; text-align: left; vertical-align: middle;">{seed_6}</td>
            <td style="padding: 8px; text-align: center; vertical-align: middle;"><b>{6}</b></td>
        </tr>
        </table>
        """, unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import requests
from PIL import Image

import sys
import os
sys.path.append(os.getcwd())
from functions import load_data

# sklearn for hypothetical model predictions
from sklearn.linear_model import LogisticRegression, LinearRegression

# ------------------------------
# PAGE CONFIGURATION & STYLING
# ------------------------------
st.set_page_config(page_title="Matchup Predictor", layout="wide")
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

# =========================
# 📦 DATA LOADING FUNCTIONS
# =========================

game_data, season_data, current, map = load_data()
current["startDate"] = pd.to_datetime(current["startDate"])

# ===== Keep preprocessing consistent with the single-team page =====
# Set future game results as "TBD"
if "completed" in current.columns and "result" in current.columns:
    current.loc[~current["completed"], "result"] = "TBD"

# Ensure week is string for game_data
if "week" in game_data.columns:
    game_data["week"] = game_data["week"].astype(str)

# Set postseason weeks to "Playoffs"
if "seasonType" in game_data.columns and "week" in game_data.columns:
    game_data.loc[game_data["seasonType"] == "postseason", "week"] = "Playoffs"

# Set start time as "TBD" when unknown
if "startTimeTBD" in current.columns and "startTime" in current.columns:
    current.loc[current["startTimeTBD"].astype(bool), "startTime"] = "TBD"

@st.cache_data
def load_team_list():
    return sorted(current["team"].unique())

@st.cache_data
def load_team_colors():
    color_dict = dict(zip(map["cfb_name"], map["Color1"]))
    return color_dict

@st.cache_data
def load_team_logos():
    return dict(zip(map["cfb_name"], map["Logo"]))

@st.cache_data
def load_team_ranks():
    df = current.copy()
    if "teamRank" in df.columns:
        # Get the MOST RECENT row per team (rank may be NaN if currently unranked)
        latest = (
            df.sort_values("startDate", ascending=False)
              .drop_duplicates(subset=["team"])
              .copy()
        )
        # Coerce to numeric; keep NaN for unranked
        latest["teamRank"] = pd.to_numeric(latest["teamRank"], errors="coerce")
        return dict(zip(latest["team"], latest["teamRank"]))
    return {}

@st.cache_data
def build_game_picker_df(current_df: pd.DataFrame) -> pd.DataFrame:
    df = current_df.copy()
    df["startDate"] = pd.to_datetime(df["startDate"])  # ensure datetime
    # Prefer away/neutral perspective rows for clarity
    if "location" in df.columns:
        df = df[df["location"].isin(["Away", "Neutral"])].copy()
    # Human-readable label
    def _mk_label(r):
        loc = r.get("location", "?")
        try:
            d = pd.to_datetime(r["startDate"]).strftime('%Y-%m-%d')
        except Exception:
            d = str(r.get("startDate", "?"))
        return f"{d} — {r['team']} @ {r['opponent']} ({loc})"
    df["label"] = df.apply(_mk_label, axis=1)
    # Carry useful columns when present
    keep = [
        "label", "team", "opponent", "location", "startDate", "startTime",
        "winProb", "predictedPoints", "predictedOpponentPoints",
        "teamRank", "opponentRank", "venue",
    ]
    keep = [c for c in keep if c in df.columns]
    df["row_index"] = df.index
    return df[keep + ["row_index"]]

team_colors = load_team_colors()
team_logos = load_team_logos()
team_ranks = load_team_ranks()
game_picker_df = build_game_picker_df(current)

# =========================
# 🤖 Functions
# =========================

def _fetch_logo_image(team: str):
    url = team_logos.get(team)
    if not url or (isinstance(url, float) and np.isnan(url)):
        return None
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGBA")
        return np.array(img)
    except Exception:
        return None
    

def fmt_team_with_rank(team: str, explicit_rank=None):
    # Prefer explicit rank from a selected schedule row
    if explicit_rank is not None and not (pd.isna(explicit_rank)):
        try:
            return f"{team} #{int(explicit_rank)}"
        except Exception:
            return f"{team} #{explicit_rank}"
    # Fallback to latest known rank
    r = team_ranks.get(team)
    if r is not None and not (pd.isna(r)):
        try:
            return f"{team} #{int(r)}"
        except Exception:
            return f"{team} #{r}"
    return team

# Helper to break ties by nudging +1 toward the predicted winner
def _tiebreak_scores(away_int: int, home_int: int, win_prob: float):
    """If rounded scores tie, bump +1 toward the side with higher win_prob.
    `win_prob` is the probability the AWAY team wins.
    """
    try:
        wp = float(win_prob)
    except Exception:
        wp = np.nan
    if away_int != home_int:
        return away_int, home_int
    # If we can't read wp, default to away +1 to avoid ties deterministically
    if np.isnan(wp):
        return away_int + 1, home_int
    # Favor the predicted winner
    if wp >= 0.5:
        return away_int + 1, home_int
    else:
        return away_int, home_int + 1

# =========================
# 🧠 MODEL HELPERS (used only for Hypothetical mode)
# =========================

def _prep_location_dummies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "location" in out.columns:
        loc = (
            out["location"].astype(str)
            .str.strip()
            .str.lower()
            .map({
                "home": "Home",
                "away": "Away",
                "neutral": "Neutral",
                "neutral site": "Neutral",
                "home/away": "Away",
            })
            .fillna("Neutral")
        )
        loc = pd.Categorical(loc, categories=["Home", "Away", "Neutral"], ordered=False)
        dums = pd.get_dummies(loc, prefix="location", drop_first=False)
        for c in ["location_Home", "location_Away", "location_Neutral"]:
            if c not in dums.columns:
                dums[c] = 0
        out = pd.concat([out.drop(columns=["location"]), dums], axis=1)
    else:
        out["location_Home"] = 0
        out["location_Away"] = 0
        out["location_Neutral"] = 1
    return out

@st.cache_data
def train_hypo_models(game_df: pd.DataFrame):
    df = game_df.copy()
    # ---- Win model (binary) ----
    win_df = df[df["result"].isin(["Win", "Loss"])].dropna(subset=["teamPregameRating"])  # require rating
    win_df["opponentPregameRating"] = win_df["opponentPregameRating"].fillna(-35)
    win_df = _prep_location_dummies(win_df)
    win_df["target"] = (win_df["result"] == "Win").astype(int)
    win_features = [
        "teamPregameRating",
        "opponentPregameRating",
        "location_Home",
        "location_Away",
        "location_Neutral",
    ]
    X_win = win_df[win_features]
    y_win = win_df["target"]
    # Clean: drop or impute NaNs/Infs before fitting
    X_win = X_win.replace([np.inf, -np.inf], np.nan)
    valid_mask = X_win.notna().all(axis=1)
    X_win = X_win.loc[valid_mask]
    y_win = y_win.loc[valid_mask]
    win_model = LogisticRegression(solver="lbfgs", max_iter=1000)
    win_model.fit(X_win, y_win)

    # ---- Score models (regression) ----
    sc_df = df.dropna(subset=["teamPoints", "opponentPoints", "teamPregameRating"]).copy()
    sc_df["opponentPregameRating"] = sc_df["opponentPregameRating"].fillna(-35)
    sc_df = _prep_location_dummies(sc_df)
    score_features = [
        "teamPregameRating",
        "opponentPregameRating",
        "location_Home",
        "location_Away",
        "location_Neutral",
    ]
    X_sc = sc_df[score_features]
    y_team = sc_df["teamPoints"]
    y_opp = sc_df["opponentPoints"]
    # Clean: drop or impute NaNs/Infs before fitting
    X_sc = X_sc.replace([np.inf, -np.inf], np.nan)
    valid_mask_sc = X_sc.notna().all(axis=1)
    X_sc = X_sc.loc[valid_mask_sc]
    y_team = y_team.loc[valid_mask_sc]
    y_opp = y_opp.loc[valid_mask_sc]
    model_team = LinearRegression().fit(X_sc, y_team)
    model_opp = LinearRegression().fit(X_sc, y_opp)

    return win_model, model_team, model_opp, win_features, score_features

@st.cache_data
def latest_ratings_map(cur: pd.DataFrame) -> dict:
    tmp = (
        cur.dropna(subset=["teamPregameRating"]) 
           .sort_values("startDate", ascending=False)
           .drop_duplicates(subset=["team"])[["team", "teamPregameRating"]]
    )
    return dict(zip(tmp["team"], tmp["teamPregameRating"]))

# =========================
# 🖱️ USER INTERFACE
# =========================

sel_label = ""

st.title("🔮 Matchup Predictor")
st.markdown("Pick a mode, then teams or a game, and generate a social-ready graphic.")

mode = st.radio("Mode", ["Hypothetical", "Pick from Schedule"], horizontal=True)

if mode == "Hypothetical":
    team_list = [""] + load_team_list()
    col1, col2, col3 = st.columns([3, 2, 3])
    with col1:
        away_team = st.selectbox("Select Away Team", team_list, index=0, placeholder="Select away team")
    with col2:
        game_type = st.selectbox("Game Type", ["Home/Away", "Neutral Site"], index=0)
    with col3:
        home_team = st.selectbox("Select Home Team", team_list, index=0, placeholder="Select home team")
    # Date & Time selectors side-by-side
    date_col, time_col = st.columns(2)
    with date_col:
        game_date = st.date_input("Game Date", value=pd.Timestamp.today().date())
    with time_col:
        time_options = [
            "12:00 PM", "3:30 PM", "7:00 PM", "8:00 PM", "10:30 PM",
        ]
        game_time = st.selectbox("Game Time (ET)", options=time_options, index=1)
    location = "Neutral" if game_type == "Neutral Site" else "Away"
    away_display = fmt_team_with_rank(away_team)
    home_display = fmt_team_with_rank(home_team)
else:
    st.markdown("Select a game from your current season data (away-team perspective).")
    sel_label = st.selectbox("Game", [""] + game_picker_df["label"].tolist(), index=0)
    if sel_label:
        row = game_picker_df.loc[game_picker_df["label"] == sel_label].iloc[0]
        away_team = row["team"]
        home_team = row["opponent"]
        location = row.get("location", "Away")
        game_date = row["startDate"].date()
        game_time = row["startTime"]
        away_display = fmt_team_with_rank(away_team, row.get("teamRank"))
        home_display = fmt_team_with_rank(home_team, row.get("opponentRank"))
    else:
        away_team = ""
        home_team = ""
        location = "Away"
        game_date = pd.Timestamp.today().date()
        game_time = "00:00"
        away_display = away_team
        home_display = home_team

# =========================
# 🚨 DUPLICATE CHECK
# =========================

if away_team == home_team:
    st.warning("Please select two different teams to simulate a matchup.")
    st.stop()

# =========================
# 📈 MAKE PREDICTION (Schedule uses `current`; Hypothetical uses trained models)
# =========================

# Safe extract helper for schedule rows
def _extract_vals_from_series(s: pd.Series):
    def _f(x):
        try:
            return float(x)
        except Exception:
            return np.nan
    return _f(s.get("winProb")), _f(s.get("predictedPoints")), _f(s.get("predictedOpponentPoints"))

win_prob = 0.5
pred_team_score = np.nan
pred_opp_score = np.nan
row_actual = None

if mode == "Pick from Schedule" and sel_label:
    # Use the selected schedule row directly from `current`
    row = game_picker_df.loc[game_picker_df["label"] == sel_label].iloc[0]
    row_actual = current.loc[row["row_index"]] if "row_index" in row else None
    if row_actual is not None:
        wp, tp, op = _extract_vals_from_series(row_actual)
    else:
        wp, tp, op = _extract_vals_from_series(row)
    if not np.isnan(wp):
        win_prob = wp
    if not np.isnan(tp):
        pred_team_score = tp
    if not np.isnan(op):
        pred_opp_score = op
else:
    # Hypothetical: train models on historical games and apply to a constructed row
    win_model, model_team, model_opp, win_features, score_features = train_hypo_models(game_data)
    ratings_dict = latest_ratings_map(current)
    team_rating = ratings_dict.get(away_team, np.nan)
    opp_rating = ratings_dict.get(home_team, np.nan)

    if pd.isna(team_rating) or pd.isna(opp_rating):
        st.warning("Missing ratings for one or both teams — using 50% and blank scores.")
    else:
        inp = pd.DataFrame([
            {
                "teamPregameRating": float(team_rating),
                "opponentPregameRating": float(opp_rating),
                "location": location,
            }
        ])
        inp = _prep_location_dummies(inp)
        for c in win_features:
            if c not in inp.columns:
                inp[c] = 0
        for c in score_features:
            if c not in inp.columns:
                inp[c] = 0
        Xw = inp[win_features]
        Xs = inp[score_features]
        win_prob = float(win_model.predict_proba(Xw)[:, 1][0])
        pred_team_score = float(model_team.predict(Xs)[0])
        pred_opp_score = float(model_opp.predict(Xs)[0])

# =========================
# 📋 DISPLAY RESULTS
# =========================

st.markdown(f"### 🧮 {away_team} at {home_team}")

left_color = team_colors.get(away_team, "#333333")
right_color = team_colors.get(home_team, "#333333")

fig, ax = plt.subplots(figsize=(8, 10))  # Portrait
ax.axis("off")

# Away team
ax.text(0.5, 0.9, f"{away_display}", ha="center", va="center",
        fontsize=24, color="#111", fontweight="bold", transform=ax.transAxes)

# Away logo
away_logo_img = _fetch_logo_image(away_team)
if away_logo_img is not None:
    away_logo_w, away_logo_h = 0.2, 0.2
    away_logo_bottom = 0.84 - away_logo_h/2
    away_logo_ax = ax.inset_axes([0.0, away_logo_bottom, away_logo_w, away_logo_h], transform=ax.transAxes)
    away_logo_ax.imshow(away_logo_img)
    away_logo_ax.axis('off')

# VS
ax.text(0.5, 0.84, "VS", ha="center", va="center",
        fontsize=22, color="#333", fontweight="bold", transform=ax.transAxes)

# Home logo
home_logo_img = _fetch_logo_image(home_team)
if home_logo_img is not None:
    home_logo_w, home_logo_h = 0.2, 0.2
    home_logo_bottom = 0.84 - home_logo_h/2
    home_logo_ax = ax.inset_axes([0.82, home_logo_bottom, home_logo_w, home_logo_h], transform=ax.transAxes)
    home_logo_ax.imshow(home_logo_img)
    home_logo_ax.axis('off')

# Home team
ax.text(0.5, 0.78, f"{home_display}", ha="center", va="center",
        fontsize=24, color="#111", fontweight="bold", transform=ax.transAxes)

# Date + location under header
if location.lower() == "neutral":
    loc_display = "Neutral Site"
else:
    loc_display = f"@ {home_team}"

ax.text(
    0.5, 0.72,
    f"{pd.to_datetime(game_date).strftime('%b %d, %Y')} • {game_time} ET  •  {loc_display}",
    ha="center", va="center", fontsize=14, color="#555", transform=ax.transAxes
)

venue_val = None
if mode == "Pick from Schedule" and 'sel_label' in locals() and sel_label:
    if "venue" in row.index and pd.notna(row["venue"]):
        venue_val = str(row["venue"])

if venue_val:
    ax.text(0.5, 0.66, venue_val, ha="center", va="center",
            fontsize=13, color="#666", transform=ax.transAxes)


# Donut pie for win probabilities
sizes = [win_prob, 1 - win_prob]
colors = [left_color, right_color]
# Center the donut within the main axes using axes-relative coords
pie_w, pie_h = 1.1, 0.55
pie_left = 0.5 - pie_w/2 +.01
pie_ax = ax.inset_axes([pie_left, 0.1, pie_w, pie_h], transform=ax.transAxes)

wedges, _ = pie_ax.pie(
    sizes,
    startangle=90,
    colors=colors,
    counterclock=True,
    wedgeprops={"width": 0.35, "edgecolor": "white"},
)
pie_ax.set_aspect('equal')
pie_ax.axis('off')

# Center score text like "14-31" (avoid ties by nudging winner +1)
#if np.isnan(pred_team_score) or np.isnan(pred_opp_score):
#    center_score = "—"
#else:
#    away_int = int(round(pred_team_score))
#    home_int = int(round(pred_opp_score))
#    away_int, home_int = _tiebreak_scores(away_int, home_int, win_prob)
#    center_score = f"{away_int}-{home_int}"
#pie_ax.text(0, 0, center_score, ha="center", va="center", fontsize=28, fontweight="bold")

# Load the winning team's logo
if win_prob >= 0.5:
    winning_logo = away_logo_img
else:
    winning_logo = home_logo_img

# Place the logo at the center of the donut
if winning_logo is not None:
    winning_logo_w, winning_logo_h = 0.25, 0.25
    winning_logo_bottom = 0.375 - winning_logo_h/2
    winning_logo_ax = ax.inset_axes([0.385, winning_logo_bottom, winning_logo_w, winning_logo_h], transform=ax.transAxes)
    winning_logo_ax.imshow(winning_logo)
    winning_logo_ax.axis('off')

# Labels on each side of donut
pie_ax.text(-1.1, 0, f"{away_team}\n{win_prob*100:.1f}%", ha="right", va="center", fontsize=12, color=left_color)
pie_ax.text(1.1, 0, f"{home_team}\n{(1 - win_prob)*100:.1f}%", ha="left", va="center", fontsize=12, color=right_color)

# Subtitle / attribution (just beneath the donut)
ax.text(0.51, 0.1, "Prediction Generated by @BG.Analytics", ha="center", va="top", fontsize=12, color="#777", transform=ax.transAxes)

st.pyplot(fig, use_container_width=False)

# Download button
buf = BytesIO()
fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
buf.seek(0)
st.download_button(
    label="⬇️ Download Graphic (PNG)",
    data=buf,
    file_name=f"{away_team}_vs_{home_team}_{pd.to_datetime(game_date).strftime('%Y%m%d')}.png",
    mime="image/png",
)
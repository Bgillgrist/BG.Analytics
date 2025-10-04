import sys
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

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

# =========================
# 📦 DATA LOADING HELPERS
# =========================
@st.cache_data(show_spinner=False)
def _load_current_games() -> pd.DataFrame:
    """Attempt to load the in-season games DataFrame from the project's
    shared `functions.py`. Falls back gracefully if shapes change.
    Expected columns (will auto-map when possible):
      - team, opponent, date, week, conference, opponentConference
      - completed (bool), result ("Win"/"Loss")
      - winProb (0-1), predSpread (float)
      - predictedPoints, predictedOpponentPoints (optional)
      - pointsFor, pointsAgainst (actual) — variety of aliases supported
    """
    # Make project root importable
    sys.path.append(os.getcwd())
    try:
        from functions import load_data  # type: ignore
    except Exception as e:  # pragma: no cover
        st.error(f"Could not import load_data from functions.py: {e}")
        return pd.DataFrame()

    try:
        data = load_data()
        # Common return shapes seen in this project:
        # 1) (game_data, season_data, current, mapping)
        # 2) (game_data, season_data, current)
        # 3) (current,) or a single DataFrame
        if isinstance(data, tuple):
            # Prefer `current` if present; else take the first DataFrame that has expected columns
            df_candidates = [d for d in data if isinstance(d, pd.DataFrame)]
            # Try the one explicitly named `current` if tuple unpack supported
            current_df = None
            if len(data) >= 3 and isinstance(data[2], pd.DataFrame):
                current_df = data[2]
            if current_df is not None:
                return current_df.copy()
            if df_candidates:
                return df_candidates[0].copy()
            return pd.DataFrame()
        elif isinstance(data, pd.DataFrame):
            return data.copy()
        else:
            return pd.DataFrame()
    except Exception as e:  # pragma: no cover
        st.error(f"load_data() raised an error: {e}")
        return pd.DataFrame()


# =========================
# 🧼 STANDARDIZATION (using your exact schema)
# =========================

def prepare_frame(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if df.empty:
        return df

    # Dates (from startDate)
    df["date"] = pd.to_datetime(df["startDate"], errors="coerce")

    # Week
    df["week"] = pd.to_numeric(df["week"], errors="coerce")

    # Conferences (team/opponent)
    df["conference"] = df["teamConference"]
    df["opponentConference"] = df["opponentConference"]

    # Actual points & totals
    df["pointsFor"] = pd.to_numeric(df["teamPoints"], errors="coerce")
    df["pointsAgainst"] = pd.to_numeric(df["opponentPoints"], errors="coerce")
    df["actualMOV"] = df["pointsFor"] - df["pointsAgainst"]
    df["actualTotal"] = df["pointsFor"] + df["pointsAgainst"]

    # Actual win (from result)
    df["actualWin"] = (df["result"].astype(str).str.lower() == "win").astype(int)

    # Conference game flag already provided; if missing, infer
    if "conferenceGame" not in df.columns:
        df["conferenceGame"] = np.where(df["conference"].eq(df["opponentConference"]), True, False)

    # Probabilities and predictions
    df["winProb"] = pd.to_numeric(df["winProb"], errors="coerce").clip(0.001, 0.999)
    df["predictedPoints"] = pd.to_numeric(df["predictedPoints"], errors="coerce")
    df["predictedOpponentPoints"] = pd.to_numeric(df["predictedOpponentPoints"], errors="coerce")
    df["predTotal"] = pd.to_numeric(df["predTotal"], errors="coerce")
    df["predSpread"] = pd.to_numeric(df["predSpread"], errors="coerce")

    # Derive Home/Away from `location` so POV can always be Home
    if "homeAway" not in df.columns and "location" in df.columns:
        df["homeAway"] = (
            df["location"].astype(str).str.strip().str.lower()
            .map({"home": "Home", "away": "Away", "neutral": "Neutral"})
            .fillna("Unknown")
        )

    df = df[df["opponentClassification"]!='fcs']

    return df


# =========================
# 📊 METRICS & SUMMARIES
# =========================
# --- Interpretable metrics ---
def accuracy(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    if not np.any(mask):
        return np.nan
    preds = (y_prob[mask] >= threshold).astype(int)
    return float((preds == y_true[mask]).mean())


def avg_prob_error(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs(y_prob[mask] - y_true[mask])))

def mae(arr: np.ndarray) -> float:
    arr = arr[~np.isnan(arr)]
    return float(np.mean(np.abs(arr))) if arr.size else np.nan


def rmse(arr: np.ndarray) -> float:
    arr = arr[~np.isnan(arr)]
    return float(np.sqrt(np.mean(arr ** 2))) if arr.size else np.nan


def summarize(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    def _agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "Games": len(g),
                "Avg WinProb": g["winProb"].mean(skipna=True),
                "Win %": g["actualWin"].mean(skipna=True),
                "Accuracy": accuracy(g["actualWin"].to_numpy(), g["winProb"].to_numpy()),
                "Avg Prob Error": avg_prob_error(g["actualWin"].to_numpy(), g["winProb"].to_numpy()),
                "MAE MOV": mae((g["predSpread"] - g["actualMOV"]).to_numpy()),
                "RMSE MOV": rmse((g["predSpread"] - g["actualMOV"]).to_numpy()),
                "MAE Total": mae((g["predTotal"] - g["actualTotal"]).to_numpy()),
                "RMSE Total": rmse((g["predTotal"] - g["actualTotal"]).to_numpy()),
            }
        )

    out = (
        df.groupby(group_cols, dropna=False)
        .apply(_agg)
        .reset_index()
        .sort_values("Games", ascending=False)
    )
    return out


# =========================
# 📥 DATA
# =========================
raw_df = _load_current_games()
df = prepare_frame(raw_df)

st.title("👀 Performance Results")

if df.empty:
    st.warning("No data available. Make sure `functions.load_data()` returns your current-season games.")
    st.stop()

# Only completed games are evaluable
comp = df[df["completed"] == True].copy()

# -------- Deduplicate to game-level --------
# 1) Use an existing game id column if available (prioritize 'id')
_gid_cols = [c for c in ["id", "gameID", "Game_ID", "game_id"] if c in comp.columns]
if _gid_cols:
    gid_col = _gid_cols[0]
    comp["_gid"] = comp[gid_col].astype(str)
else:
    # Fallback: canonical key = date + sorted(team/opponent)
    comp["_teamA"] = comp["team"].astype(str)
    comp["_teamB"] = comp["opponent"].astype(str)
    comp["_match"] = comp.apply(lambda r: "||".join(sorted([r["_teamA"], r["_teamB"]])), axis=1)
    comp["_date_str"] = pd.to_datetime(comp["date"], errors="coerce").astype(str)
    comp["_gid"] = comp["_date_str"] + "::" + comp["_match"]

# 3) Prefer HOME side for consistency; else take a stable first row
if "homeAway" in comp.columns:
    # 0 = Home (best), 1 = Neutral, 2 = Away, 3 = other/unknown
    pref_map = {"Home": 0, "Neutral": 1, "Away": 2}
    comp["_pref"] = comp["homeAway"].map(pref_map).fillna(3).astype(int)
    comp = (
        comp.sort_values(["_gid", "_pref", "date", "team"])  # stable sort; keeps Home when available
            .groupby("_gid", as_index=False)
            .first()
    )
    comp.drop(columns=["_pref"], inplace=True, errors="ignore")
else:
    # Fallback if no homeAway available anywhere
    comp = (
        comp.sort_values(["_gid", "date", "team"])  # stable sort
            .groupby("_gid", as_index=False)
            .first()
    )

if comp.empty:
    st.info("No completed games yet to evaluate.")
    st.stop()

# =========================
# 🔎 FILTERS (Moved to very top)
# =========================
with st.expander("Filters", expanded=True):
    left, mid, right = st.columns([2, 2, 2])

    all_confs = sorted(pd.unique(pd.concat([comp["conference"], comp["opponentConference"]]).dropna()))
    selected_confs = left.multiselect("Conferences (team perspective)", options=all_confs, default=all_confs)

    weeks = sorted(comp["week"].dropna().astype(int).unique().tolist())
    if weeks:
        w1, w2 = int(min(weeks)), int(max(weeks))
    else:
        w1, w2 = 1, 15

    if w1 == w2:
        # Only one completed week; avoid slider min==max error
        week_range = (w1, w2)
        mid.markdown(f"**Week range:** Week {w1}")
    else:
        week_range = mid.slider("Week range", min_value=w1, max_value=w2, value=(w1, w2), step=1)

    # Home/Away if available
    hoa = "homeAway" if "homeAway" in comp.columns else None
    hoa_opts = ["All", "Home", "Away"] if hoa else ["All"]
    selected_hoa = right.selectbox("Venue", hoa_opts, index=0)

# Apply filters
mask = comp["conference"].isin(selected_confs)
mask &= comp["week"].between(week_range[0], week_range[1])
if hoa and selected_hoa != "All":
    mask &= comp[hoa].str.title().eq(selected_hoa)

comp = comp[mask].copy()

# =========================
# 🗺️ UPSET MAP (Win Prob vs Actual MOV, full width, after Filters)
# =========================
# Predictions at 0.5 threshold
comp["_pred"] = (comp["winProb"] >= 0.5).astype(int)

upset_df = comp[["winProb", "actualMOV", "team", "opponent", "week", "_pred", "actualWin"]].dropna(subset=["winProb", "actualMOV"]).copy()

if not upset_df.empty:
    upset_df["Outcome"] = np.where(upset_df["_pred"] == upset_df["actualWin"], "Correct", "Incorrect")
    # Outliers
    wrong_df = upset_df[upset_df["Outcome"] == "Incorrect"].copy()
    wrong_df["x_dist"] = np.abs(wrong_df["winProb"] - 0.5)
    extreme_wrong = wrong_df.sort_values("x_dist", ascending=False).head(4)
    fig = px.scatter(
        upset_df,
        x="winProb",
        y="actualMOV",
        color="Outcome",
        hover_data=["team", "opponent", "week"],
        title="Upset Map: Win Prob vs Actual MOV",
        labels={"winProb": "Predicted Win Probability", "actualMOV": "Actual Margin of Victory"},
        color_discrete_map={"Correct": "green", "Incorrect": "red"},
        height=650,
    )
    # Reference lines
    fig.add_vline(x=0.5, line_dash="dash")
    fig.add_hline(y=0, line_dash="dash")
# Add text for quadrants
    fig.add_annotation(x=0.10, y=60, text="<b>Underdog Won</b>", showarrow=False,font=dict(size=16, color="darkred"))
    fig.add_annotation(x=0.75, y=60, text="<b>Favorite Won</b>", showarrow=False,font=dict(size=16, color="darkgreen"))
    fig.add_annotation(x=0.1, y=-30, text="<b>Favorite Won</b>", showarrow=False,font=dict(size=16, color="darkgreen"))
    fig.add_annotation(x=0.75, y=-30, text="<b>Underdog Won</b>", showarrow=False,font=dict(size=16, color="darkred"))
# Add markers
    fig.add_trace(go.Scatter(
        x=extreme_wrong["winProb"],
        y=extreme_wrong["actualMOV"],
        mode="markers+text",
        name="Farthest Wrong",
        marker=dict(size=9, color="red", symbol="x"),
        text=extreme_wrong["team"] + " vs " + extreme_wrong["opponent"],
        textposition=[
            "top center" if wp < 0.5 else "bottom center"
            for wp in extreme_wrong["winProb"]
        ],
        textfont=dict(size=11, color="black"),
        showlegend=False
    ))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough data for Upset Map.")


# =========================
# 📊 Confusion Matrix & Game-by-Game Table (side-by-side)
# =========================
cm_left, cm_right = st.columns([1, 2])

# Confusion matrix counts
TP = int(((comp["actualWin"] == 1) & (comp["_pred"] == 1)).sum())
TN = int(((comp["actualWin"] == 0) & (comp["_pred"] == 0)).sum())
FP = int(((comp["actualWin"] == 0) & (comp["_pred"] == 1)).sum())
FN = int(((comp["actualWin"] == 1) & (comp["_pred"] == 0)).sum())

cm_z = [[TN, FP], [FN, TP]]
cm_x = ["Pred: Loss", "Pred: Win"]
cm_y = ["Actual: Loss", "Actual: Win"]
cm_text = [[f"TN\n{TN}", f"FP\n{FP}"], [f"FN\n{FN}", f"TP\n{TP}"]]

cm_fig = go.Figure(data=go.Heatmap(
    z=cm_z,
    x=cm_x,
    y=cm_y,
    colorscale="Blues",
    showscale=True,
    hovertemplate="%{y} / %{x}<br>Count: %{z}<extra></extra>",
))
# Add annotations for counts
for i, ylab in enumerate(cm_y):
    for j, xlab in enumerate(cm_x):
        cm_fig.add_annotation(x=xlab, y=ylab, text=cm_text[i][j], showarrow=False, font=dict(color="black"))

cm_fig.update_layout(
    title="Confusion Matrix (Threshold = 0.5)",
    xaxis_title="Predicted Class",
    yaxis_title="Actual Class",
    height=420,
    margin=dict(l=10, r=10, t=50, b=10),
)
cm_left.plotly_chart(cm_fig, use_container_width=True)

# Per-game table with row highlighting by correctness
per_game = comp[[
    "date", "team", "opponent", "winProb", "_pred", "actualWin", "predSpread", "actualMOV"
]].copy()

# Nicely formatted columns
per_game["Date"] = pd.to_datetime(per_game["date"], errors="coerce").dt.date
per_game["Team"] = per_game["team"].astype(str)
per_game["Opponent"] = per_game["opponent"].astype(str)
per_game["Win Prob"] = (per_game["winProb"] * 100).round(1)
per_game["Pred"] = np.where(per_game["_pred"] == 1, "Win", "Loss")
per_game["Actual"] = np.where(per_game["actualWin"] == 1, "Win", "Loss")
per_game["Correct"] = per_game["_pred"] == per_game["actualWin"]
per_game["Pred MOV"] = per_game["predSpread"].round(1)
per_game["Actual MOV"] = per_game["actualMOV"].round(1)

per_game_display = per_game[[
    "Date", "Team", "Opponent", "Win Prob", "Pred", "Actual", "Correct", "Pred MOV", "Actual MOV"
]].copy()

# Style rows: green if correct, red if incorrect
def _row_color(r: pd.Series):
    color = "background-color: #e6ffed" if r.get("Correct", False) else "background-color: #ffecec"
    return [color] * len(r)

styled = per_game_display.style.apply(_row_color, axis=1).format({
    "Win Prob": "{:.1f}%",
})

with cm_right:
    st.markdown("**Game-by-Game Predictions**")
    st.dataframe(styled, use_container_width=True, hide_index=True)

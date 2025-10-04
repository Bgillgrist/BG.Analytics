import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import streamlit as st
from collections import Counter
from typing import Dict, List, Tuple, Optional

#Data Import Functions

@st.cache_data
def load_data():
    game_data = pd.read_csv("data/game_data.csv")
    season_data = pd.read_csv("data/season_data.csv")
    current = pd.read_csv("data/game_data_current.csv")
    map = pd.read_csv("data/name_mapping.csv")
    return game_data, season_data, current, map

#Win Percentage Model Training Functions

@st.cache_resource
def train_win_model(games_df):
    df = games_df.copy()
    df = df[df["result"].isin(["Win", "Loss"])]
    df["opponentPregameRating"] = df["opponentPregameRating"].fillna(-35)
    df = df.dropna(subset=["teamPregameRating"])
    df["target"] = (df["result"] == "Win").astype(int)

    df = pd.get_dummies(df, columns=["location"], drop_first=False)
    expected_features = ["teamPregameRating", "opponentPregameRating", "location_Home", "location_Away", "location_Neutral"]
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    X = df[expected_features]
    y = df["target"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model, expected_features

def apply_win_model(upcoming_df, model, features):
    df = upcoming_df.copy()
    df["opponentPregameRating"] = df["opponentPregameRating"].fillna(-35)
    df["teamPregameRating"] = df["teamPregameRating"].fillna(0)
    df = pd.get_dummies(df, columns=["location"], drop_first=False)

    for col in features:
        if col not in df.columns:
            df[col] = 0

    return model.predict_proba(df[features])[:, 1]

def normalize_win_probs(df):
    df = df.copy()

    if "id" not in df.columns:
        raise ValueError("Game_ID column required for probability normalization.")

    # Skip normalization for FCS opponents
    df["skip_norm"] = df["opponentClassification"].str.lower() == "fcs"

    def conditional_norm(group):
        if group["skip_norm"].any():
            return group["winProb"]
        total = group["winProb"].sum()
        return group["winProb"] / total if total > 0 else group["winProb"]

    df["winProb"] = df.groupby("id", group_keys=False).apply(conditional_norm)

    df.drop(columns="skip_norm", inplace=True)

    return df

# Score prediction Training Functions

def train_total_model(games_df):
    df = games_df.copy()
    df["totalScore"] = df["teamPoints"] + df["opponentPoints"]
    df["isFCS"] = (
        df.get("opponentClassification", "").astype(str).str.strip().str.lower().eq("fcs").astype(int)
    )
    df = df.dropna(subset=["totalScore", "teamPregameRating", "opponentPregameRating", "week", "isFCS", "conferenceGame"])
    df["opponentPregameRating"] = df["opponentPregameRating"].fillna(-35)

    # One-hot encode location
    df = pd.get_dummies(df, columns=["location"], drop_first=False)
    total_features = ["teamPregameRating", "opponentPregameRating", "isFCS", "conferenceGame", "week"]
    for col in total_features:
        if col not in df.columns:
            df[col] = 0

    X = df[total_features]
    Y = df["totalScore"]

    model_total = LinearRegression().fit(X, Y)

    return model_total, total_features

def train_spread_model(games_df):
    df = games_df.copy()
    df["isFCS"] = (
        df.get("opponentClassification", "").astype(str).str.strip().str.lower().eq("fcs").astype(int)
    )
    df = df.dropna(subset=["MOV", "teamPregameRating", "opponentPregameRating", "location", "isFCS"])
    df["opponentPregameRating"] = df["opponentPregameRating"].fillna(-35)

    # One-hot encode location
    df = pd.get_dummies(df, columns=["location"], drop_first=False)
    spread_features = ["teamPregameRating", "opponentPregameRating", "location_Home", "location_Away", "location_Neutral", "isFCS"]
    for col in spread_features:
        if col not in df.columns:
            df[col] = 0

    X = df[spread_features]
    Y = df["MOV"]

    model_spread = LinearRegression().fit(X, Y)

    return model_spread, spread_features

def apply_score_model(current_df, model_total, model_spread, total_features, spread_features):
    df = current_df.copy()

    # Keep original 'location' for later use
    original_location = df["location"] if "location" in df.columns else None

    df["opponentPregameRating"] = df["opponentPregameRating"].fillna(-35)
    df["teamPregameRating"] = df["teamPregameRating"].fillna(0)

    df = pd.get_dummies(df, columns=["location"], drop_first=False)
    for col in total_features:
        if col not in df.columns:
            df[col] = 0

    total_X = df[total_features]
    spread_X = df[spread_features]
    df["predSpread"] = model_spread.predict(spread_X).round(2)
    df["predTotal"] = model_total.predict(total_X).round(2)

    # Restore location column if it was dropped
    if "location" not in df.columns and original_location is not None:
        df["location"] = original_location

    df["predictedPoints"] = df["predTotal"] - ((df["predTotal"] - df["predSpread"])/2)
    df["predictedOpponentPoints"] = ((df["predTotal"] - df["predSpread"])/2)

    return df



# MAJOR SEASON SIMULATION FUNCTION

@st.cache_data
def simulate_full_season(
    df: pd.DataFrame,
    _model: LogisticRegression,
    features: list,
    num_simulations: int = 5000,
    coef_mov: float = 0.08237826714816475,
) -> pd.DataFrame:
    """
    Monte Carlo full-season simulator for College Football.

    Expected columns in `df` (one row per team-game):
      - team: str
      - teamConference: str
      - completed: bool
      - result: "Win" or "Loss" (only for completed games)
      - conferenceGame: bool (True if counts toward conference standings)
      - winProb: float in [0,1] for incomplete games
      - potentialPlayoffPoints: float (EV base for a *win* in that game)
      - predictedPoints: float (team points prediction for that game)
      - predictedOpponentPoints: float (opp points prediction for that game)
      - actualPlayoffPoints: float (for completed games; else NaN or 0)

    Notes:
      - Selection uses “Top‑5 conference champions by Playoff Points” as auto‑bids,
        then “next 7 highest Playoff Points overall” as at‑large.
      - Seeds 1–12 are assigned by Playoff Points rank (1 = best).
      - Conference title game participants are the top two teams in each conference
        by simulated conference wins (tie broken by higher Playoff Points that season).
      - Title game winner is sampled using a simple Bradley–Terry style win prob
        based on their *season Playoff Points* as a strength proxy.

    Returns a DataFrame with one row per team and columns:
      team, conference,
      wins_mode, wins_avg, losses_mode, losses_avg,
      conf_wins_mode, conf_wins_avg,
      wins_floor_95, wins_ceil_95, wins_floor_75, wins_ceil_75,
      pp_avg,
      conf_title_game_pct, conf_champ_pct, playoff_pct,
      auto_bid_pct, at_large_pct,
      seed_avg, seed_mode, top4_bye_pct,
      current_wins, current_losses,
      num_sims
    """

    # ----------- Precompute fixed (non-random) “current” values -----------
    teams = df["team"].unique()
    conferences_map: Dict[str, str] = (
        df[["team", "teamConference"]]
        .drop_duplicates()
        .set_index("team")["teamConference"]
        .to_dict()
    )

    # Current W/L are derived from completed games and do NOT vary by simulation
    cur_wins = (
        df[(df["completed"]) & (df["result"] == "Win")]
        .groupby("team")
        .size()
        .reindex(teams, fill_value=0)
        .to_dict()
    )
    cur_losses = (
        df[(df["completed"]) & (df["result"] == "Loss")]
        .groupby("team")
        .size()
        .reindex(teams, fill_value=0)
        .to_dict()
    )

    # Split completed vs incomplete for speed
    completed = df[df["completed"]].copy()
    incomplete = df[~df["completed"]].copy()

    # Completed-game playoff points (fixed)
    base_points_completed = (
        completed.groupby("team")["actualPlayoffPoints"]
        .sum()
        .reindex(teams, fill_value=0.0)
        .to_dict()
    )

    # For incomplete games, precompute MOV component and keep needed fields
    if not incomplete.empty:
        inc_games = incomplete[[
            "team", "conferenceGame", "winProb", "potentialPlayoffPoints",
            "predictedPoints", "predictedOpponentPoints"
        ]].copy()
        inc_games["mov_pred"] = inc_games["predictedPoints"] - inc_games["predictedOpponentPoints"]
        inc_games["pp_if_win"] = inc_games["potentialPlayoffPoints"] + coef_mov * inc_games["mov_pred"]
    else:
        inc_games = pd.DataFrame(columns=[
            "team", "conferenceGame", "winProb", "pp_if_win"
        ])

    # Keep a list of teams per conference (exclude FBS Independents from conf title)
    conf_teams: Dict[str, List[str]] = {}
    for t in teams:
        conf = conferences_map.get(t, "Unknown")
        conf_teams.setdefault(conf, []).append(t)

    # ----------- Storage for simulation tallies -----------
    tallies: Dict[str, Dict[str, list]] = {
        t: {
            "wins": [],
            "losses": [],
            "conf_wins": [],
            "pp": [],
            "title_game": [],   # 1 if reached conference title game
            "conf_champ": [],   # 1 if won conference title game
            "playoff": [],      # 1 if in final 12
            "auto_bid": [],     # 1 if auto-bid
            "at_large": [],     # 1 if at-large
            "seed": []          # integer seed 1..12 for entries; np.nan if not in
        }
        for t in teams
    }

    # ----------- Main Monte Carlo loop -----------
    for _ in range(num_simulations):
        # Per-sim running season stats
        wins = {t: cur_wins.get(t, 0) for t in teams}
        losses = {t: cur_losses.get(t, 0) for t in teams}
        conf_wins = {t: int(0) for t in teams}
        pp = {t: base_points_completed.get(t, 0.0) for t in teams}

        # Count conference wins already banked from completed games
        if not completed.empty:
            comp_conf = completed[completed["conferenceGame"] == True]
            comp_conf_w = comp_conf[comp_conf["result"] == "Win"].groupby("team").size()
            for t, c in comp_conf_w.items():
                conf_wins[t] += int(c)

        # Simulate the remaining games
        if not inc_games.empty:
            # Vectorized Bernoulli draws per row
            draws = np.random.random(len(inc_games))
            won_mask = draws < inc_games["winProb"].values

            # Wins/Losses
            win_counts = inc_games.loc[won_mask].groupby("team").size()
            loss_counts = inc_games.loc[~won_mask].groupby("team").size()
            for t, c in win_counts.items():
                wins[t] += int(c)
            for t, c in loss_counts.items():
                losses[t] += int(c)

            # Conference wins on simulated wins
            conf_mask = (inc_games["conferenceGame"] == True) & won_mask
            conf_win_counts = inc_games.loc[conf_mask].groupby("team").size()
            for t, c in conf_win_counts.items():
                conf_wins[t] += int(c)

            # Playoff points for simulated wins
            pp_add = inc_games.loc[won_mask].groupby("team")["pp_if_win"].sum()
            for t, v in pp_add.items():
                pp[t] += float(v)

        # ---------- Determine conference title game participants ----------
        # Rule: top two by conf_wins; tie-breaker by higher Playoff Points (pp)
        # Independents cannot play a title game
        title_pairs: Dict[str, Tuple[str, str]] = {}
        for conf, members in conf_teams.items():
            if conf in ("FBS Independents", "Independents", "Unknown"):
                continue
            if len(members) < 2:
                continue

            sub = pd.DataFrame({
                "team": members,
                "conf_wins": [conf_wins[t] for t in members],
                "pp": [pp[t] for t in members],
            }).sort_values(["conf_wins", "pp"], ascending=[False, False])

            t1, t2 = sub.iloc[0]["team"], sub.iloc[1]["team"]
            title_pairs[conf] = (t1, t2)
            tallies[t1]["title_game"].append(1)
            tallies[t2]["title_game"].append(1)

        # Teams that did not reach a title game this sim
        for t in teams:
            if len(tallies[t]["title_game"]) < _ + 1:
                tallies[t]["title_game"].append(0)

        # ---------- Simulate conference title games ----------
        conf_champs: List[str] = []
        for conf, (t1, t2) in title_pairs.items():
            matchup_df = pd.DataFrame([{
                "teamPregameRating": df[df["team"] == t1]["teamPregameRating"].dropna().iloc[-1] if not df[df["team"] == t1]["teamPregameRating"].dropna().empty else 0.0,
                "opponentPregameRating": df[df["team"] == t2]["teamPregameRating"].dropna().iloc[-1] if not df[df["team"] == t2]["teamPregameRating"].dropna().empty else 0.0,
                "location": "Neutral",
            }])
            p1 = float(apply_win_model(matchup_df, _model, features)[0])
            if np.random.random() < p1:
                champ, runner = t1, t2
            else:
                champ, runner = t2, t1
            conf_champs.append(champ)
            tallies[champ]["conf_champ"].append(1)
            if len(tallies[runner]["conf_champ"]) < _ + 1:
                tallies[runner]["conf_champ"].append(0)

        # Non-participants (or participants who already got a 1/0 above):
        for t in teams:
            if len(tallies[t]["conf_champ"]) < _ + 1:
                tallies[t]["conf_champ"].append(0)

        # ---------- Select playoff field (12 teams) ----------
        season_df = pd.DataFrame({
            "team": teams,
            "pp": [pp[t] for t in teams],
            "is_champ": [1 if t in conf_champs else 0 for t in teams],
        })

        # Top-5 champs by pp = auto bids
        champ_board = season_df[season_df["is_champ"] == 1].sort_values("pp", ascending=False)
        auto_bids = champ_board.head(5)["team"].tolist()

        # Remaining spots by pp (at-large)
        remaining = season_df[~season_df["team"].isin(auto_bids)].sort_values("pp", ascending=False)
        at_large = remaining.head(7)["team"].tolist()

        field = auto_bids + at_large  # 12 teams

        # Seed by pp rank among the 12 (1 = best)
        seeded = (
            season_df[season_df["team"].isin(field)]
            .sort_values("pp", ascending=False)
            .reset_index(drop=True)
        )
        seed_map = {team: i + 1 for i, team in enumerate(seeded["team"].tolist())}

        # Record outcomes for this sim
        for t in teams:
            tallies[t]["wins"].append(wins[t])
            tallies[t]["losses"].append(losses[t])
            tallies[t]["conf_wins"].append(conf_wins[t])
            tallies[t]["pp"].append(pp[t])

            in_field = t in field
            is_auto = t in auto_bids
            is_at_large = t in at_large
            tallies[t]["playoff"].append(1 if in_field else 0)
            tallies[t]["auto_bid"].append(1 if is_auto else 0)
            tallies[t]["at_large"].append(1 if is_at_large else 0)
            tallies[t]["seed"].append(seed_map[t] if in_field else np.nan)

    # ----------- Aggregate to summary table -----------
    def mode_int(arr: List[int]) -> int:
        # Handles multi-modal by taking the smallest (stable for wins/seed)
        c = Counter(arr)
        top = max(c.values())
        candidates = [k for k, v in c.items() if v == top]
        return int(min(candidates))

    rows = []
    for t in teams:
        W = np.array(tallies[t]["wins"])
        L = np.array(tallies[t]["losses"])
        CW = np.array(tallies[t]["conf_wins"])
        PP = np.array(tallies[t]["pp"])
        TG = np.array(tallies[t]["title_game"])
        CC = np.array(tallies[t]["conf_champ"])
        PO = np.array(tallies[t]["playoff"])
        AB = np.array(tallies[t]["auto_bid"])
        AL = np.array(tallies[t]["at_large"])
        SD = np.array(tallies[t]["seed"])

        # Seed stats only over entries where in field
        seed_vals = SD[~np.isnan(SD)].astype(int)
        seed_avg = float(seed_vals.mean()) if seed_vals.size else np.nan
        seed_mode = mode_int(seed_vals.tolist()) if seed_vals.size else np.nan

        rows.append({
            "team": t,
            "conference": conferences_map.get(t, "Unknown"),

            "wins_mode": mode_int(W.tolist()),
            "wins_avg": float(W.mean()),
            "losses_mode": mode_int(L.tolist()),
            "losses_avg": float(L.mean()),
            "conf_wins_mode": mode_int(CW.tolist()),
            "conf_wins_avg": float(CW.mean()),

            "wins_floor_95": float(np.percentile(W, 2.5)),
            "wins_ceil_95": float(np.percentile(W, 97.5)),
            "wins_floor_75": float(np.percentile(W, 12.5)),
            "wins_ceil_75": float(np.percentile(W, 87.5)),

            "pp_avg": float(PP.mean()),

            "conf_title_game_pct": 100.0 * float(TG.mean()),
            "conf_champ_pct": 100.0 * float(CC.mean()),
            "playoff_pct": 100.0 * float(PO.mean()),
            "auto_bid_pct": round(100.0 * float(AB.mean()), 2),
            "at_large_pct": round(100.0 * float(AL.mean()), 2),

            "seed_avg": seed_avg,
            "seed_mode": seed_mode,
            "top4_bye_pct": 100.0 * (float((seed_vals <= 4).mean()) if seed_vals.size else 0.0),

            "current_wins": cur_wins.get(t, 0),
            "current_losses": cur_losses.get(t, 0),

            "num_sims": int(num_simulations),
        })

    summary = pd.DataFrame(rows)

    # Nice ordering for dashboard use
    col_order = [
        "team", "conference",
        "wins_mode", "wins_avg", "losses_mode", "losses_avg",
        "conf_wins_mode", "conf_wins_avg",
        "wins_floor_95", "wins_ceil_95", "wins_floor_75", "wins_ceil_75",
        "pp_avg",
        "conf_title_game_pct", "conf_champ_pct", "playoff_pct",
        "auto_bid_pct", "at_large_pct",
        "seed_avg", "seed_mode", "top4_bye_pct",
        "current_wins", "current_losses",
        "num_sims",
    ]
    return summary[col_order]
import sys
import os
sys.path.append(os.getcwd())
from functions import load_data, simulate_full_season, train_win_model

import streamlit as st
import pandas as pd
import numpy as np
import itertools
from typing import List, Dict, Tuple, Optional

# ==============================
# ⚙️ PAGE CONFIG & STYLES
# ==============================
st.set_page_config(page_title="🎯 Playoff Paths — CFB Dashboard", layout="wide")

st.markdown(
    """
    <style>
        .block-container { padding-top: 2.2rem; }
        .statcard { background:#f7f9fc; border:1px solid #e6e9ef; border-radius:10px; padding:10px 14px; }
        .statnum { font-size:1.4rem; font-weight:700; }
        .statlbl { color:#5b677a; font-size:0.9rem; }
        .muted   { color:#6b7280; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="background-color:#002D62; padding:4px 10px; border-radius:10px; margin-bottom:10px;">
        <h1 style="color:white; text-align:center;">🎯 Path to the Playoffs</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# 🔁 Controls — Rerun & Cache
# ------------------------------
col_r1, col_r2 = st.columns([1, 8])
with col_r1:
    if st.button("🔁 Re-run (fresh sims)", help="Clears cache and reruns the page with fresh randomness"):
        try:
            st.cache_data.clear()
            st.cache_resource.clear()
        except Exception:
            pass
        st.rerun()

# ==============================
# 📦 DATA LOADING
# ==============================
with st.spinner("Loading data…"):
    try:
        # Preferred signature: game_data, season_data, current, map
        game_data, season_data, current, mapping = load_data()
        win_model, win_model_features = train_win_model(game_data)
    except Exception:
        # Fallbacks if your load_data signature differs
        out = load_data()
        # Try best-effort unpack
        if isinstance(out, (list, tuple)) and len(out) >= 3:
            game_data, season_data, current = out[:3]
            mapping = out[3] if len(out) > 3 else pd.DataFrame()
        else:
            # As last resort assume just current
            current = out
            game_data = pd.DataFrame()
            season_data = pd.DataFrame()
            mapping = pd.DataFrame()

# ==============================
# 🔧 Helpers
# ==============================
COEF_MOV = 0.08237826714816475  # must match your simulator default


def first_col(df: pd.DataFrame, candidates: List[str], default: Optional[str] = None) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return default


def remaining_games(df: pd.DataFrame, team: str) -> pd.DataFrame:
    """Return remaining games for team with helper columns."""
    opp_col = first_col(df, ["opponent", "Opponent", "opponentTeam", "opp"])
    date_col = first_col(df, ["startDate", "date", "Date"])  # allow NaN
    ha_col = first_col(df, ["homeAway", "Home/Away", "home_away"])  # allow NaN
    conf_col = first_col(df, ["conferenceGame", "confGame", "isConference"])  # bool-ish

    need_cols = [c for c in ["team", opp_col, "completed", "winProb",
                             "potentialPlayoffPoints", "predictedPoints",
                             "predictedOpponentPoints", date_col, ha_col, conf_col] if c]
    sub = df.loc[df["team"] == team, need_cols].copy()
    sub = sub[sub["completed"] != True].copy()

    # Derived
    sub["mov_pred"] = (sub.get("predictedPoints", 0) - sub.get("predictedOpponentPoints", 0)).astype(float)
    sub["pp_if_win"] = sub.get("potentialPlayoffPoints", 0).astype(float) + COEF_MOV * sub["mov_pred"]
    sub["winProb"] = sub["winProb"].fillna(0.5).clip(0, 1)

    # Labels
    if ha_col and opp_col:
        sub["label"] = np.where(sub[ha_col].astype(str).str[:1].str.lower().isin(["h", "n"]),
                                 "vs " + sub[opp_col].astype(str),
                                 "at " + sub[opp_col].astype(str))
    elif opp_col:
        sub["label"] = sub[opp_col].astype(str)
    else:
        sub["label"] = "Game"

    if date_col in sub:
        try:
            sub["when"] = pd.to_datetime(sub[date_col]).dt.date.astype(str)
        except Exception:
            sub["when"] = sub[date_col].astype(str)
    else:
        sub["when"] = "TBD"

    if conf_col in sub:
        sub["conf"] = sub[conf_col].astype(bool)
    else:
        sub["conf"] = True

    sub["row_id"] = sub.index  # use row index as unique key within this session
    return sub[["row_id", "label", "when", "conf", "winProb", "pp_if_win"]]


def _locate_opponent_rows(df: pd.DataFrame, row: pd.Series) -> List[int]:
    """Best-effort find opponent's mirror rows for a given team-row.
    Uses team/opponent/date if present; returns list of row indices (can be empty)."""
    opp_col = first_col(df, ["opponent", "Opponent", "opponentTeam", "opp"]) or "opponent"
    date_col = first_col(df, ["startDate", "date", "Date"])  # optional

    if opp_col not in df.columns:
        return []

    team = row.get("team")
    opp = row.get(opp_col)

    mask = (df["team"] == opp) & (df.get(opp_col, pd.Series(index=df.index)) == team)
    if date_col and date_col in df.columns and isinstance(row.get(date_col, None), (str, pd.Timestamp)):
        try:
            # Use date match if available to avoid double matches
            mask = mask & (pd.to_datetime(df[date_col]) == pd.to_datetime(row[date_col]))
        except Exception:
            pass

    return df[mask].index.tolist()


def apply_scenario(
    df: pd.DataFrame,
    team: str,
    force_wins: List[int],  # list of row indices in df for the *team's* rows
    force_losses: List[int],
    nudge: float = 0.0,     # add to team winProb for its remaining games
) -> pd.DataFrame:
    """Apply scenario locks and nudges to a copy of the season df.
    - For forced wins/losses we set completed=True, result, and actualPlayoffPoints accordingly.
    - We also best-effort flip the opponent mirror rows to the opposite result.
    - For nudges, we adjust *only the selected team's* remaining games.
    """
    mod = df.copy()
    opp_col = first_col(mod, ["opponent", "Opponent", "opponentTeam", "opp"]) or "opponent"
    date_col = first_col(mod, ["startDate", "date", "Date"])  # optional

    # Ensure required columns exist
    for c in ["completed", "result", "actualPlayoffPoints"]:
        if c not in mod.columns:
            mod[c] = np.nan

    # Nudge: only remaining games for selected team
    rem_mask = (mod["team"] == team) & (mod["completed"] != True)
    if abs(nudge) > 1e-9 and "winProb" in mod.columns:
        mod.loc[rem_mask, "winProb"] = mod.loc[rem_mask, "winProb"].fillna(0.5) + float(nudge)
        mod.loc[rem_mask, "winProb"] = mod.loc[rem_mask, "winProb"].clip(0, 1)

    # Helper to compute pp_if_win for an arbitrary row
    def row_pp_if_win(s: pd.Series) -> float:
        base = float(s.get("potentialPlayoffPoints", 0) or 0)
        mov = float((s.get("predictedPoints", 0) or 0) - (s.get("predictedOpponentPoints", 0) or 0))
        return base + COEF_MOV * mov

    # Force wins/losses on selected team's rows
    for rid in force_wins:
        if rid in mod.index:
            s = mod.loc[rid]
            if s.get("team") != team:
                continue
            mod.at[rid, "completed"] = True
            mod.at[rid, "result"] = "Win"
            mod.at[rid, "actualPlayoffPoints"] = row_pp_if_win(s)
            # Mirror opponent rows → Loss
            for oid in _locate_opponent_rows(mod, s):
                mod.at[oid, "completed"] = True
                mod.at[oid, "result"] = "Loss"
                mod.at[oid, "actualPlayoffPoints"] = 0.0

    for rid in force_losses:
        if rid in mod.index:
            s = mod.loc[rid]
            if s.get("team") != team:
                continue
            mod.at[rid, "completed"] = True
            mod.at[rid, "result"] = "Loss"
            mod.at[rid, "actualPlayoffPoints"] = 0.0
            # Mirror opponent rows → Win
            for oid in _locate_opponent_rows(mod, s):
                mod.at[oid, "completed"] = True
                mod.at[oid, "result"] = "Win"
                mod.at[oid, "actualPlayoffPoints"] = row_pp_if_win(mod.loc[oid])

    return mod


def quick_impact_score(row: pd.Series) -> float:
    # Heuristic: higher win prob × playoff point haul → bigger swing
    return float(row.get("winProb", 0.5)) * float(row.get("pp_if_win", 0.0))


def enumerate_minimal_paths(
    base_df: pd.DataFrame,
    team: str,
    rem_tbl: pd.DataFrame,
    max_candidates: int = 8,
    max_k: int = 3,
    top_n: int = 12,
    target: str = "Either",  # "Auto-bid" | "At-large" | "Either"
    sims_per_eval: int = 1200,
) -> pd.DataFrame:
    """Enumerate minimal sets of forced wins that produce high playoff probability.
    Minimal means no proper subset already qualifies.
    Returns a DataFrame of paths with: size, games, path_prob, playoff_pct, auto_bid_pct, at_large_pct, seed_mode.
    """
    # Rank candidate games by impact
    cand = rem_tbl.copy()
    cand["impact"] = cand.apply(quick_impact_score, axis=1)
    cand = cand.sort_values("impact", ascending=False).head(max_candidates)

    # Baseline for conditioning
    base_summary = simulate_full_season(base_df, win_model, win_model_features, num_simulations=sims_per_eval)
    base_row = base_summary.loc[base_summary["team"] == team]
    if base_row.empty:
        return pd.DataFrame(columns=["Path", "Size", "Games", "Path Prob", "Playoff %", "Auto-bid %", "At-large %", "Seed Mode"])  # safety

    # Helper to check target criterion
    def meets_target(row: pd.Series) -> bool:
        if target == "Auto-bid":
            return row.get("auto_bid_pct", 0.0) >= 50.0
        if target == "At-large":
            return row.get("at_large_pct", 0.0) >= 50.0
        return row.get("playoff_pct", 0.0) >= 50.0

    # Storage
    paths: List[Dict] = []
    minimal_sets: List[set] = []

    # Iterate subset sizes
    idx_list = cand["row_id"].tolist()
    label_map = cand.set_index("row_id")["label"].to_dict()
    prob_map = cand.set_index("row_id")["winProb"].to_dict()

    for k in range(1, max_k + 1):
        for combo in itertools.combinations(idx_list, k):
            sset = set(combo)
            # Prune: if any existing minimal set is subset of this, skip
            if any(ms.issubset(sset) for ms in minimal_sets):
                continue

            # Apply scenario: force these as wins
            mod_df = apply_scenario(base_df, team=team, force_wins=list(combo), force_losses=[], nudge=0.0)
            sim = simulate_full_season(mod_df, win_model, win_model_features, num_simulations=sims_per_eval)
            row = sim.loc[sim["team"] == team].iloc[0].to_dict()

            if meets_target(row):
                minimal_sets.append(sset)
                games_str = ", ".join([label_map[x] for x in combo])
                path_prob = float(np.prod([prob_map[x] for x in combo]))
                paths.append({
                    "Path": f"{k}-win path",
                    "Size": k,
                    "Games": games_str,
                    "Path Prob": 100.0 * path_prob,
                    "Playoff %": row.get("playoff_pct", 0.0),
                    "Auto-bid %": row.get("auto_bid_pct", 0.0),
                    "At-large %": row.get("at_large_pct", 0.0),
                    "Seed Mode": row.get("seed_mode", np.nan),
                })

                if len(paths) >= top_n:
                    break
        if len(paths) >= top_n:
            break

    if not paths:
        return pd.DataFrame(columns=["Path", "Size", "Games", "Path Prob", "Playoff %", "Auto-bid %", "At-large %", "Seed Mode"])  # empty

    df_paths = pd.DataFrame(paths).sort_values(["Size", "Path Prob"], ascending=[True, False])
    return df_paths


def rank_outside_game_flips(
    base_df: pd.DataFrame,
    team: str,
    summary_df: pd.DataFrame,
    top_pairs: int = 10,
    sims_per_eval: int = 800,
) -> pd.DataFrame:
    """Find outside (non-team) remaining games whose flip to an upset helps the team.
    Returns: Game, Flip (fav loses), Our Δ Playoff %, Favored Team (approx), Baseline Fav WinProb
    """
    opp_col = first_col(base_df, ["opponent", "Opponent", "opponentTeam", "opp"]) or "opponent"
    date_col = first_col(base_df, ["startDate", "date", "Date"])  # optional

    # Candidate = remaining games where neither side is our team; deduplicate pairs
    rem = base_df[(base_df["completed"] != True)].copy()
    rem = rem[(rem["team"] != team) & (rem[opp_col] != team)].copy()

    # Merge in team playoff % to rank contenders
    use_cols = ["team", "playoff_pct", "pp_avg"]
    team_stats = summary_df[use_cols].copy()
    rem = rem.merge(team_stats, on="team", how="left", suffixes=("", "_team"))

    # Create a pair key and keep one row per game pair
    def pair_key(r):
        a, b = sorted([str(r.get("team")), str(r.get(opp_col))])
        if date_col and date_col in rem.columns:
            try:
                d = pd.to_datetime(r.get(date_col)).date().isoformat()
            except Exception:
                d = str(r.get(date_col))
            return f"{a}__{b}__{d}"
        return f"{a}__{b}"

    rem["pair_key"] = rem.apply(pair_key, axis=1)
    rem = rem.sort_values(["pair_key", "playoff_pct"], ascending=[True, False])
    rem = rem.drop_duplicates("pair_key", keep="first").copy()

    # Approx favored team = this row's team if winProb >= 0.5 else opponent side
    rem["fav_team"] = np.where(rem.get("winProb", pd.Series(0.5, index=rem.index)).fillna(0.5) >= 0.5, rem["team"], rem[opp_col])
    rem["fav_winprob"] = rem.get("winProb", pd.Series(0.5, index=rem.index)).fillna(0.5).clip(0, 1)

    # Rank by opponent contention (higher of either side's playoff %)
    # Merge opponent playoff %
    opp_stats = summary_df[use_cols].rename(columns={"team": opp_col, "playoff_pct": "opp_playoff_pct", "pp_avg": "opp_pp_avg"})
    rem = rem.merge(opp_stats, on=opp_col, how="left")
    rem["contender_score"] = rem[["playoff_pct", "opp_playoff_pct"]].fillna(0.0).max(axis=1)
    rem = rem.sort_values(["contender_score", "fav_winprob"], ascending=[False, False]).head(top_pairs)

    # Evaluate flips
    results = []
    for _, r in rem.iterrows():
        # Force favored team to LOSE this pair (best-effort: modify both sides)
        mod = base_df.copy()
        # team side
        mask_a = (mod["team"] == r["fav_team"]) & (mod[opp_col] == r[opp_col])
        # opponent side mirror
        mask_b = (mod["team"] == r[opp_col]) & (mod[opp_col] == r["fav_team"]) if opp_col in mod.columns else pd.Series(False, index=mod.index)

        for idx in mod[mask_a].index.tolist():
            mod.at[idx, "completed"] = True
            mod.at[idx, "result"] = "Loss"
            mod.at[idx, "actualPlayoffPoints"] = 0.0
        for idx in mod[mask_b].index.tolist():
            # compute pp_if_win for opponent mirror row
            base = float(mod.at[idx, "potentialPlayoffPoints"]) if "potentialPlayoffPoints" in mod.columns else 0.0
            mov = float(mod.at[idx, "predictedPoints"]) - float(mod.at[idx, "predictedOpponentPoints"]) if ("predictedPoints" in mod.columns and "predictedOpponentPoints" in mod.columns) else 0.0
            mod.at[idx, "completed"] = True
            mod.at[idx, "result"] = "Win"
            mod.at[idx, "actualPlayoffPoints"] = base + COEF_MOV * mov

        sim = simulate_full_season(mod, win_model, win_model_features, num_simulations=sims_per_eval)
        row = sim.loc[sim["team"] == team]
        if row.empty:
            continue
        new_playoff = float(row.iloc[0]["playoff_pct"])  # after flip
        base_playoff = float(summary_df.loc[summary_df["team"] == team].iloc[0]["playoff_pct"]) if not summary_df.empty else np.nan
        results.append({
            "Game": f"{r.get('team')} vs {r.get(opp_col)}",
            "Flip": "Fav loses",
            "Favored Team": r.get("fav_team"),
            "Fav WinProb": 100.0 * float(r.get("fav_winprob", 0.5)),
            "Our Δ Playoff %": new_playoff - base_playoff,
        })

    if not results:
        return pd.DataFrame(columns=["Game", "Flip", "Favored Team", "Fav WinProb", "Our Δ Playoff %"])  # empty

    df_imp = pd.DataFrame(results).sort_values("Our Δ Playoff %", ascending=False)
    return df_imp

# ==============================
# 🎛️ TOP CONTROLS
# ==============================
teams = sorted(current["team"].unique()) if not current.empty else []
selected = st.selectbox("Select a team", teams, index=0 if teams else None)

with st.expander("Scenario builder", expanded=True):
    left, mid, right = st.columns([2, 2, 1])
    with left:
        rem_tbl = remaining_games(current, selected) if selected else pd.DataFrame()
        st.caption("**Your remaining games** (click to force results)")
        if not rem_tbl.empty:
            # Checkboxes to force wins/losses
            force_wins_ids = []
            force_losses_ids = []
            for _, r in rem_tbl.iterrows():
                c1, c2 = st.columns([4, 2])
                with c1:
                    st.write(f"**{r['label']}** — {r['when']}")
                with c2:
                    win = st.checkbox("Win", key=f"win_{int(r['row_id'])}")
                    loss = st.checkbox("Loss", key=f"loss_{int(r['row_id'])}")
                if win and not loss:
                    force_wins_ids.append(int(r["row_id"]))
                if loss and not win:
                    force_losses_ids.append(int(r["row_id"]))
        else:
            st.info("No remaining games detected.")

    with mid:
        target = st.selectbox("Target path", ["Either", "Auto-bid", "At-large"], help="Which path definition should minimal paths satisfy?")
        deep = st.toggle("Deep search (slower)", value=False, help="Explore more games and larger combos")
        max_candidates = 10 if deep else 8
        max_k = 4 if deep else 3
        sims_per_eval = 1500 if deep else 1000
        nudge = st.slider("Performance nudge to our win probabilities", min_value=-0.15, max_value=0.15, value=0.0, step=0.01)

    with right:
        st.markdown("&nbsp;")
        if st.button("Apply scenario & recompute"):
            st.session_state["paths_apply"] = True
            st.rerun()

# Apply scenario to base df for this page session (so impacts/paths use locks/nudge)
scenario_df = apply_scenario(current, selected, force_wins=force_wins_ids if selected else [], force_losses=force_losses_ids if selected else [], nudge=nudge)

# ==============================
# 📊 BASELINE SNAPSHOT TILES
# ==============================
with st.spinner("Running baseline simulation…"):
    sim_summary = simulate_full_season(scenario_df, win_model, win_model_features, num_simulations=3000)

row = sim_summary.loc[sim_summary["team"] == selected]
if row.empty:
    st.warning("Selected team not found in simulation summary.")
    st.stop()
team_row = row.iloc[0]

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.markdown("<div class='statcard'><div class='statnum'>%.1f%%</div><div class='statlbl'>Playoff %%</div></div>" % team_row["playoff_pct"], unsafe_allow_html=True)
with c2:
    st.markdown("<div class='statcard'><div class='statnum'>%.1f%%</div><div class='statlbl'>Auto‑Bid %%</div></div>" % team_row["auto_bid_pct"], unsafe_allow_html=True)
with c3:
    st.markdown("<div class='statcard'><div class='statnum'>%.1f%%</div><div class='statlbl'>At‑Large %%</div></div>" % team_row["at_large_pct"], unsafe_allow_html=True)
with c4:
    st.markdown("<div class='statcard'><div class='statnum'>%.1f%%</div><div class='statlbl'>Conf Champ %%</div></div>" % team_row["conf_champ_pct"], unsafe_allow_html=True)
with c5:
    st.markdown("<div class='statcard'><div class='statnum'>%.1f</div><div class='statlbl'>Seed Avg</div></div>" % (team_row["seed_avg"] if not np.isnan(team_row["seed_avg"]) else 0), unsafe_allow_html=True)
with c6:
    st.markdown("<div class='statcard'><div class='statnum'>%d–%d</div><div class='statlbl'>Current W–L</div></div>" % (team_row["current_wins"], team_row["current_losses"]), unsafe_allow_html=True)

st.markdown("---")

# ==============================
# 🧭 CONTROL YOUR OWN DESTINY (Minimal Paths)
# ==============================
st.subheader("Control Your Own Destiny — Minimal Winning Paths")
left_paths, right_help = st.columns([3, 2])

with left_paths:
    rem_tbl = remaining_games(scenario_df, selected)
    if rem_tbl.empty:
        st.info("No remaining games to generate paths.")
    else:
        with st.spinner("Enumerating minimal paths…"):
            paths_df = enumerate_minimal_paths(
                base_df=scenario_df,
                team=selected,
                rem_tbl=rem_tbl,
                max_candidates=max_candidates,
                max_k=max_k,
                top_n=12,
                target=target,
                sims_per_eval=sims_per_eval,
            )
        if paths_df.empty:
            st.warning("No minimal paths found that reach ≥50% under current settings. Try Deep Search or adjust the nudge/locks.")
        else:
            # Pretty columns
            show_df = paths_df.copy()
            show_df["Path Prob"] = show_df["Path Prob"].map(lambda x: f"{x:.1f}%")
            show_df["Playoff %"] = show_df["Playoff %"].map(lambda x: f"{x:.1f}%")
            show_df["Auto-bid %"] = show_df["Auto-bid %"].map(lambda x: f"{x:.1f}%")
            show_df["At-large %"] = show_df["At-large %"].map(lambda x: f"{x:.1f}%")
            st.dataframe(show_df, use_container_width=True, hide_index=True)
            st.download_button("Download paths CSV", paths_df.to_csv(index=False).encode("utf-8"), file_name=f"{selected}_minimal_paths.csv")

with right_help:
    st.caption("**How this works**")
    st.markdown(
        """
        We rank your remaining games by **impact** (win prob × playoff‑point haul), then search for the
        smallest sets of wins that push your playoff odds to **≥50%** (according to fresh sims), honoring any
        locks/nudges above. We stop at 12 paths by default.

        • *Target path* lets you focus on **Auto‑bid** (Top‑5 champs), **At‑large**, or either.
        
        • *Deep search* increases both the candidate pool and the max combo size.
        """
    )

st.markdown("---")

# ==============================
# 📣 OTHER TEAMS' LOSSES THAT HELP YOU
# ==============================
st.subheader("Other Teams’ Losses That Help You")
with st.spinner("Evaluating single-game flips…"):
    outside_df = rank_outside_game_flips(
        base_df=scenario_df, team=selected, summary_df=sim_summary,
        top_pairs=10 if not deep else 14, sims_per_eval=700 if not deep else 1000
    )

if outside_df.empty:
    st.info("No impactful outside flips identified under current settings.")
else:
    show_out = outside_df.copy()
    show_out["Fav WinProb"] = show_out["Fav WinProb"].map(lambda x: f"{x:.0f}%")
    show_out["Our Δ Playoff %"] = show_out["Our Δ Playoff %"].map(lambda x: f"{x:+.1f} pp")
    st.dataframe(show_out, use_container_width=True, hide_index=True)
    st.download_button("Download rooting guide CSV", outside_df.to_csv(index=False).encode("utf-8"), file_name=f"{selected}_rooting_guide.csv")

st.markdown("---")

# ==============================
# 🗓️ SCHEDULE PLAN — US & OTHERS
# ==============================
st.subheader("Schedule Plan")
colA, colB = st.columns(2)

with colA:
    st.caption("**Your remaining games** (impact & required flags)")
    rem_tbl2 = remaining_games(scenario_df, selected)
    if not rem_tbl2.empty:
        # Mark if in any minimal path
        if 'paths_df' in locals() and not paths_df.empty:
            req_counts: Dict[str, int] = {}
            for games in paths_df["Games"].tolist():
                for g in [x.strip() for x in games.split(",") if x.strip()]:
                    req_counts[g] = req_counts.get(g, 0) + 1
            rem_tbl2["Required?"] = rem_tbl2["label"].map(lambda x: "Yes" if req_counts.get(x, 0) > 0 else "No")
            rem_tbl2["In Top Paths %"] = rem_tbl2["label"].map(lambda x: 100.0 * req_counts.get(x, 0) / max(1, len(paths_df)))
        else:
            rem_tbl2["Required?"] = "No"
            rem_tbl2["In Top Paths %"] = 0.0
        tmp = rem_tbl2.copy()
        tmp["Impact"] = tmp.apply(quick_impact_score, axis=1)
        tmp = tmp.rename(columns={"label": "Game", "when": "When", "conf": "Conf?", "winProb": "WinProb", "pp_if_win": "PP_if_Win"})
        tmp["WinProb"] = (100 * tmp["WinProb"]).round(0)
        tmp["PP_if_Win"] = tmp["PP_if_Win"].round(2)
        tmp["Impact"] = tmp["Impact"].round(2)
        st.dataframe(tmp[["Game", "When", "Conf?", "WinProb", "PP_if_Win", "Impact", "Required?", "In Top Paths %"]], use_container_width=True, hide_index=True)
    else:
        st.info("No remaining games.")

with colB:
    st.caption("**Key outside games** (estimated benefit if favorite loses)")
    if outside_df.empty:
        st.info("No outside games ranked yet.")
    else:
        st.dataframe(show_out, use_container_width=True, hide_index=True)

st.markdown("---")

# ==============================
# 🧮 SELECTION VIEW — THRESHOLDS & CONTEXT
# ==============================
st.subheader("Selection View — Thresholds & Context")

# Approximate PP threshold to be in Top‑12 by PP (EV view)
pp_sorted = sim_summary.sort_values("pp_avg", ascending=False).reset_index(drop=True)
pp_thresh = float(pp_sorted.iloc[11]["pp_avg"]) if len(pp_sorted) >= 12 else float(pp_sorted.iloc[-1]["pp_avg"]) if not pp_sorted.empty else np.nan

cL, cR = st.columns(2)
with cL:
    st.markdown(f"**Projected PP (EV)**: `{team_row['pp_avg']:.2f}`  |  **Approx. Threshold (12th)**: `{pp_thresh:.2f}`")
    pct_to_thresh = 100.0 * (team_row['pp_avg'] / pp_thresh) if pp_thresh and pp_thresh != 0 else 0
    st.progress(min(max(pct_to_thresh / 100.0, 0.0), 1.0), text=f"{pct_to_thresh:.0f}% of threshold")

with cR:
    seed_avg = team_row.get("seed_avg", np.nan)
    seed_mode = team_row.get("seed_mode", np.nan)
    bye = team_row.get("top4_bye_pct", 0.0)
    st.markdown(f"**Seed (avg/mode)**: `{seed_avg:.1f}` / `{seed_mode}`  |  **Top‑4 Bye**: `{bye:.1f}%`")

st.caption("This threshold is a back‑of‑the‑envelope EV check (12th by `pp_avg`). Actual selection is run per‑sim in the Monte Carlo above.")

# ==============================
# ✅ FOOTER — EXPORT SUMMARY
# ==============================
st.markdown("---")
st.caption("Quick exports")
exp_left, exp_right = st.columns(2)
with exp_left:
    quick_summary = pd.DataFrame({
        "Metric": [
            "Playoff %", "Auto‑Bid %", "At‑Large %", "Conf Title Game %", "Conf Champ %",
            "Seed Avg", "Seed Mode", "Top‑4 Bye %", "Current W", "Current L", "PP Avg"
        ],
        "Value": [
            f"{team_row['playoff_pct']:.1f}%", f"{team_row['auto_bid_pct']:.1f}%", f"{team_row['at_large_pct']:.1f}%",
            f"{team_row['conf_title_game_pct']:.1f}%", f"{team_row['conf_champ_pct']:.1f}%",
            f"{team_row['seed_avg']:.1f}", f"{team_row['seed_mode']}", f"{team_row['top4_bye_pct']:.1f}%",
            int(team_row['current_wins']), int(team_row['current_losses']), f"{team_row['pp_avg']:.2f}"
        ]
    })
    st.dataframe(quick_summary, hide_index=True, use_container_width=True)

with exp_right:
    export_bundle = {
        "summary.csv": sim_summary.to_csv(index=False),
        "paths.csv": (paths_df.to_csv(index=False) if 'paths_df' in locals() and not paths_df.empty else pd.DataFrame().to_csv(index=False)),
        "rooting_guide.csv": (outside_df.to_csv(index=False) if not outside_df.empty else pd.DataFrame().to_csv(index=False)),
    }
    # Single file buttons (easiest UX)
    st.download_button("Download summary CSV", export_bundle["summary.csv"].encode("utf-8"), file_name=f"{selected}_summary.csv")
    if 'paths_df' in locals() and not paths_df.empty:
        st.download_button("Download paths CSV", export_bundle["paths.csv"].encode("utf-8"), file_name=f"{selected}_paths.csv")
    if not outside_df.empty:
        st.download_button("Download rooting guide CSV", export_bundle["rooting_guide.csv"].encode("utf-8"), file_name=f"{selected}_rooting_guide.csv")
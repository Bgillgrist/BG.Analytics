# =========================
# 📦 IMPORTS & CONFIGURATION
# =========================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from functions import load_data, train_win_model, apply_win_model, train_spread_model, train_total_model, apply_score_model, normalize_win_probs

import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import pytz
import time
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
#API_KEY = os.getenv("CFBD_API_KEY")
API_KEY = "SqExgR1dkCL6yQ4PnYStsn8z9wg5bmlryiW5vustoRNFGH8Jb/9WtYzqXugBKru8"
headers = {'Authorization': f'Bearer {API_KEY}'}

# SET CURRENT SEASON HERE!!!!!!
year = 2025

# Toggle for static TeamRankings ratings (Basically used in the offseason when the ratings are only on one date)
use_single_date_rating = False
single_date = "2025-08-21"


# =========================
# 🏈 FETCH & PREPARE GAME DATA
# =========================

def fetch_games(year):
    all_data = []
    for season_type in ['regular', 'postseason']:
        url = 'https://api.collegefootballdata.com/games'
        params = {'year': year, 'seasonType': season_type}
        response = requests.get(url, headers=headers, params=params)
        time.sleep(1.2)
        if response.status_code == 200 and 'application/json' in response.headers.get('Content-Type', ''):
            data = response.json()
            if data:
                all_data.extend(data)
    return pd.DataFrame(all_data)


def duplicate_teams(games):
    def prep(df, home=True):
        prefix = 'home' if home else 'away'
        opp_prefix = 'away' if home else 'home'
        df['team'] = df[f'{prefix}Team']
        df['opponent'] = df[f'{opp_prefix}Team']
        df['teamConference'] = df[f'{prefix}Conference']
        df['opponentConference'] = df[f'{opp_prefix}Conference']
        df['teamClassification'] = df[f'{prefix}Classification']
        df['opponentClassification'] = df[f'{opp_prefix}Classification']
        df['teamPoints'] = df[f'{prefix}Points']
        df['opponentPoints'] = df[f'{opp_prefix}Points']
        df['teamPregameElo'] = df[f'{prefix}PregameElo']
        df['opponentPregameElo'] = df[f'{opp_prefix}PregameElo']
        df['teamPostgameElo'] = df[f'{prefix}PostgameElo']
        df['opponentPostgameElo'] = df[f'{opp_prefix}PostgameElo']
        df['teamPostgameWinProb'] = df[f'{prefix}PostgameWinProbability']
        df['opponentPostgameWinProb'] = df[f'{opp_prefix}PostgameWinProbability']
        df['location'] = 'Home' if home else 'Away'
        df['location'] = df['location'].mask(df['neutralSite'], 'Neutral')
        return df

    home_df = prep(games.copy(), home=True)
    away_df = prep(games.copy(), home=False)

    for df in [home_df, away_df]:
        df['MOV'] = df['teamPoints'] - df['opponentPoints']
        df['result'] = df['MOV'].apply(lambda x: 'Win' if x > 0 else ('Loss' if x < 0 else 'Tie'))

    return pd.concat([home_df, away_df], ignore_index=True)


def process_dates(game_data):
    game_data['startDateUTC'] = pd.to_datetime(game_data['startDate'], utc=True)
    eastern = pytz.timezone('US/Eastern')
    game_data['startDateEastern'] = game_data['startDateUTC'].dt.tz_convert(eastern)
    game_data['startDate'] = game_data['startDateEastern'].dt.strftime('%Y-%m-%d')
    game_data['startTime'] = game_data['startDateEastern'].dt.strftime('%I:%M %p')
    game_data.loc[game_data['startTimeTBD'].astype(bool), 'startTime'] = "TBD"
    return game_data


# =========================
# 📊 RANKINGS & ELO HANDLING
# =========================

def fetch_rankings(year):
    all_rankings = []
    url = "https://api.collegefootballdata.com/rankings"
    response = requests.get(url, headers=headers, params={'year': year})
    time.sleep(1.2)

    if response.status_code == 200:
        try:
            data = response.json()
            if not data:
                print(f"No rankings available yet for {year}.")
                return pd.DataFrame()
            for week in data:
                for poll in week.get('polls', []):
                    for rank in poll.get('ranks', []):
                        all_rankings.append({
                            'season': week['season'],
                            'week': week['week'],
                            'seasonType': week['seasonType'],
                            'poll': poll['poll'],
                            'team': rank['school'],
                            'rank': rank['rank']
                        })
        except Exception as e:
            print(f"Error parsing rankings data: {e}")
            return pd.DataFrame()
    else:
        print(f"Failed to fetch rankings for {year}. Status code: {response.status_code}")
        return pd.DataFrame()

    rankings_df = pd.DataFrame(all_rankings)
    rankings_df['poll_priority'] = rankings_df['poll'].map({
        'Playoff Committee Rankings': 1,
        'AP Top 25': 2,
        'Coaches Poll': 3
    })
    rankings_df = rankings_df[rankings_df['poll_priority'].notna()]
    rankings_df = rankings_df.sort_values(by=['season', 'week', 'seasonType', 'team', 'poll_priority'])
    return rankings_df.drop_duplicates(subset=['season', 'week', 'seasonType', 'team'], keep='first')


def merge_rankings(game_data, rankings_df):
    if rankings_df.empty:
        print("Skipping ranking merge — no rankings available.")
        game_data['teamRank'] = np.nan
        game_data['opponentRank'] = np.nan
        return game_data
    
    game_data['week'] = game_data['week'].astype(int)
    rankings_df['week'] = rankings_df['week'].astype(int)
    # Prepare team ranking
    team_ranks = rankings_df.rename(columns={'rank': 'teamRank'})[
        ['season', 'week', 'seasonType', 'team', 'teamRank']
    ]
    # Prepare opponent ranking
    opponent_ranks = rankings_df.rename(columns={'team': 'opponent', 'rank': 'opponentRank'})[
        ['season', 'week', 'seasonType', 'opponent', 'opponentRank']
    ]
    game_data = game_data.merge(
        team_ranks,
        how='left',
        on=['season', 'week', 'seasonType', 'team']
    )
    game_data = game_data.merge(
        opponent_ranks,
        how='left',
        on=['season', 'week', 'seasonType', 'opponent']
    )
    return game_data

# =========================
# 📈 TEAMRANKINGS RATING SCRAPER
# =========================

def scrape_teamrankings_ratings(date_str, mapping_df):
    url = f"https://www.teamrankings.com/college-football/ranking/predictive-by-other?date={date_str}"
    response = requests.get(url)
    time.sleep(1.2)

    if response.status_code != 200:
        print(f"Failed to fetch data for {date_str}")
        return {}

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", class_="tr-table datatable scrollable")
    if not table:
        print(f"No table found for {date_str}")
        return {}

    rows = table.find_all("tr")[1:]
    name_map = {
        tr_name.strip().replace('\xa0', ''): cfb_name
        for cfb_name, tr_name in zip(mapping_df["cfb_name"], mapping_df["teamrankings_name"])
    }

    ratings = {}
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 3:
            continue
        tr_name = re.sub(r"\s+\(\d+-\d+\)", "", cols[1].text.strip()).replace('\xa0', '')
        try:
            rating = float(cols[2].text.strip())
        except ValueError:
            continue

        cfb_name = name_map.get(tr_name)
        if cfb_name:
            ratings[cfb_name] = rating
        else:
            print(f"Unmatched team name: {tr_name}")

    return ratings


def add_teamrankings_ratings(games_df, mapping_df):
    games_df = games_df.copy()
    games_df["teamPregameRating"] = np.nan
    games_df["opponentPregameRating"] = np.nan
    games_df["teamPostgameRating"] = np.nan
    games_df["opponentPostgameRating"] = np.nan

    all_dates = pd.to_datetime(games_df["startDate"]).sort_values().unique()
    team_rating_history = {}

    if use_single_date_rating:
        ratings = scrape_teamrankings_ratings(single_date, mapping_df)

        games_df["teamPregameRating"] = games_df["team"].map(ratings)
        games_df["opponentPregameRating"] = games_df["opponent"].map(ratings)
    else:
        for game_date in all_dates:
            date_ts = pd.Timestamp(game_date)
            print(f"fetching {game_date}")
            pregame_date = (date_ts - timedelta(days=1)).strftime("%Y-%m-%d")
            postgame_date = (date_ts + timedelta(days=1)).strftime("%Y-%m-%d")

            pre_ratings = scrape_teamrankings_ratings(pregame_date, mapping_df)
            post_ratings = scrape_teamrankings_ratings(postgame_date, mapping_df)
            team_rating_history[game_date] = pre_ratings

            mask = games_df["startDate"] == date_ts.strftime("%Y-%m-%d")
            for idx in games_df[mask].index:
                team = games_df.at[idx, "team"]
                opp = games_df.at[idx, "opponent"]

                games_df.at[idx, "teamPregameRating"] = pre_ratings.get(team)
                games_df.at[idx, "opponentPregameRating"] = pre_ratings.get(opp)

                if games_df.at[idx, "completed"]:
                    games_df.at[idx, "teamPostgameRating"] = post_ratings.get(team)
                    games_df.at[idx, "opponentPostgameRating"] = post_ratings.get(opp)
                else:
                    for past_date in reversed(all_dates[all_dates <= game_date]):
                        past_ratings = team_rating_history.get(past_date, {})
                        if pd.isna(games_df.at[idx, "teamPregameRating"]):
                            games_df.at[idx, "teamPregameRating"] = past_ratings.get(team)
                        if pd.isna(games_df.at[idx, "opponentPregameRating"]):
                            games_df.at[idx, "opponentPregameRating"] = past_ratings.get(opp)

    return games_df

def set_fcs_opponent_ratings(games_df):
    fcs_mask = games_df["opponentClassification"].str.lower() == "fcs"

    for col in ["opponentPregameRating", "opponentPostgameRating"]:
        games_df.loc[fcs_mask, col] = -30

    return games_df

# =========================
# ADD WIN PERCENTAGE COLUMN
# =========================

# Found in the functions file

# =========================
# 🏆 PLAYOFF POINTS CALCULATION
# =========================

def add_playoff_points(df):
    # === Ensure correct types ===
    df["opponentPregameRating"] = pd.to_numeric(df["opponentPregameRating"], errors="coerce")
    df["MOV"] = pd.to_numeric(df.get("MOV", 0), errors="coerce").fillna(0)

    # === Feature Engineering ===
    df["location_factor"] = df["location"].str.lower().map({
        "away": 2.0,
        "neutral": 1.5,
        "home": 1.0
    }).fillna(1.0)

    df["week_scaled"] = df["week"] / 10

    # === Model Coefficients ===
    BASE = 9.126148769748378
    COEF_LOCATION = 0.8183707391511457
    COEF_OPP_RATING = 0.20555802743091757
    COEF_WEEK = 0.5978282863498295
    COEF_MOV = 0.08237826714816475

    # === Calculate Potential Playoff Points (NO MOV) ===
    df["potentialPlayoffPoints"] = (
        BASE
        + COEF_LOCATION * df["location_factor"]
        + COEF_OPP_RATING * df["opponentPregameRating"]
        + COEF_WEEK * df["week_scaled"]
    )

    # === Calculate Actual Playoff Points (if Win) ===
    df["actualPlayoffPoints"] = df.apply(
        lambda row: row["potentialPlayoffPoints"] + COEF_MOV * row["MOV"]
        if row["completed"] and row["result"].lower() == "win"
        else 0,
        axis=1
    )

    return round(df, 4)

# =========================
# ❤️ FINAL DATA FIXES
# =========================

def fill_missing_ratings(game_data): 
        game_data = game_data.sort_values(by=['team', 'startDate'])
        game_data['teamPregameRating'] = game_data.groupby('team')['teamPregameRating'].transform(lambda x: x.ffill())

        game_data = game_data.sort_values(by=['opponent', 'startDate'])
        game_data['opponentPregameRating'] = game_data.groupby('opponent')['opponentPregameRating'].transform(lambda x: x.ffill())
        return game_data

# GET HISTORICAL DATA

game_data = pd.read_csv("data/game_data.csv")

# =========================
# 🚀 MAIN EXECUTION
# =========================

def main():
    print(f"Fetching {year} data...")
    raw_games = fetch_games(year)
    games = duplicate_teams(raw_games)
    games = process_dates(games)
    rankings = fetch_rankings(year)
    games = merge_rankings(games, rankings)
    mapping_df = pd.read_csv("data/name_mapping.csv")
    games = add_teamrankings_ratings(games, mapping_df)
    games = set_fcs_opponent_ratings(games)
    
    # Add win percentage column
    
    model, features = train_win_model(game_data)
    games["winProb"] = apply_win_model(games, model, features)

    # Add Score Predictions

    model_total, total_features = train_total_model(game_data)
    model_spread, spread_features = train_spread_model(game_data)
    games = apply_score_model(games, model_total, model_spread, total_features, spread_features)
    games = normalize_win_probs(games)

    # Add Playoff Points
    games = add_playoff_points(games)

    games = games[games['teamClassification'] == 'fbs']

    final_columns = [
        'id', 'season', 'week', 'seasonType', 'startDate', 'startTime', 'startTimeTBD', 'completed', 'team', 'opponent',
        'location', 'teamPoints', 'opponentPoints', 'result', 'MOV', 'conferenceGame', 'attendance', 'venue',
        'teamConference', 'opponentConference', 'teamClassification', 'opponentClassification',
        'teamPregameRating', 'opponentPregameRating', 'teamPostgameRating', 'opponentPostgameRating',
        'teamRank', 'opponentRank', 'notes', 'winProb', 'predictedPoints', 'predictedOpponentPoints', 'predTotal', 'predSpread',
        'potentialPlayoffPoints', 'actualPlayoffPoints'
    ]

    os.makedirs("data", exist_ok=True)
    games[final_columns].to_csv("data/game_data_current.csv", index=False)
    print("✅ Current season data saved to data/game_data_current.csv")


if __name__ == "__main__":
    main()
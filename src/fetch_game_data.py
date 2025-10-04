# =========================
# 📦 IMPORTS & CONFIGURATION
# =========================

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import pytz
import time
import os
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Configure year range for full historical data
start = 2010
end = 2025

# Load API key
load_dotenv()
API_KEY = os.getenv("CFBD_API_KEY")
headers = {'Authorization': f'Bearer {API_KEY}'}


# =========================
# 🏈 FETCH & PREPARE GAME DATA
# =========================

def fetch_games(start_year=start, end_year=end):
    all_data = []
    for year in range(start_year, end_year):
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
    return game_data


# =========================
# 📊 RANKINGS & ELO HANDLING
# =========================

def fetch_rankings(start_year=start, end_year=end):
    all_rankings = []
    for year in range(start_year, end_year):
        url = "https://api.collegefootballdata.com/rankings"
        response = requests.get(url, headers=headers, params={'year': year})
        time.sleep(1.2)
        if response.status_code == 200:
            data = response.json()
            for week in data:
                for poll in week['polls']:
                    for rank in poll['ranks']:
                        all_rankings.append({
                            'season': week['season'],
                            'week': week['week'],
                            'seasonType': week['seasonType'],
                            'poll': poll['poll'],
                            'team': rank['school'],
                            'rank': rank['rank']
                        })
    rankings_df = pd.DataFrame(all_rankings)
    rankings_df['poll_priority'] = rankings_df['poll'].map({
        'Playoff Committee Rankings': 1,
        'AP Top 25': 2
    })
    rankings_df = rankings_df[rankings_df['poll_priority'].notna()]
    rankings_df = rankings_df.sort_values(by=['season', 'week', 'seasonType', 'team', 'poll_priority'])
    return rankings_df.drop_duplicates(subset=['season', 'week', 'seasonType', 'team'], keep='first')


def merge_rankings(game_data, rankings_df):
    game_data = game_data.merge(
        rankings_df.rename(columns={'rank': 'teamRank'}),
        how='left',
        on=['season', 'week', 'seasonType', 'team']
    )
    rankings_df_opp = rankings_df.rename(columns={'team': 'opponent', 'rank': 'opponentRank'})
    game_data = game_data.merge(
        rankings_df_opp[['season', 'week', 'seasonType', 'opponent', 'opponentRank']],
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


def add_teamrankings_columns(games_df, mapping_df):
    for col in ["teamPregameRating", "opponentPregameRating", "teamPostgameRating", "opponentPostgameRating"]:
        games_df[col] = np.nan

    unique_dates = pd.to_datetime(games_df["startDate"]).sort_values().unique()

    for game_date in unique_dates:
        game_date_ts = pd.Timestamp(game_date)
        game_date_str = game_date_ts.strftime("%Y-%m-%d")

        try:
            pregame_date = (game_date_ts - timedelta(days=1)).strftime("%Y-%m-%d")
            postgame_date = (game_date_ts + timedelta(days=1)).strftime("%Y-%m-%d")

            pre_ratings = scrape_teamrankings_ratings(pregame_date, mapping_df)
            post_ratings = scrape_teamrankings_ratings(postgame_date, mapping_df)

            mask = games_df["startDate"] == game_date_str
            for idx in games_df[mask].index:
                team = games_df.at[idx, "team"]
                opponent = games_df.at[idx, "opponent"]

                games_df.at[idx, "teamPregameRating"] = pre_ratings.get(team)
                games_df.at[idx, "opponentPregameRating"] = pre_ratings.get(opponent)
                games_df.at[idx, "teamPostgameRating"] = post_ratings.get(team)
                games_df.at[idx, "opponentPostgameRating"] = post_ratings.get(opponent)

            print(f"Processed ratings for {game_date_str}")

        except Exception as e:
            print(f"Error processing {game_date_str}: {e}")

    return games_df

def set_fcs_opponent_ratings(games_df):
    fcs_mask = games_df["opponentClassification"].str.lower() == "fcs"

    for col in ["opponentPregameRating", "opponentPostgameRating"]:
        games_df.loc[fcs_mask, col] = -30

    return games_df

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
    BASE = 8.926148769748378
    COEF_LOCATION = 0.8183707391511457
    COEF_OPP_RATING = 0.29555802743091757
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
# 🚀 MAIN EXECUTION
# =========================

def main():
    print("Fetching game data...")
    raw_games = fetch_games()
    games = duplicate_teams(raw_games)
    games = process_dates(games)
    rankings = fetch_rankings()
    games = merge_rankings(games, rankings)
    mapping_df = pd.read_csv("data/name_mapping.csv")
    games = add_teamrankings_columns(games, mapping_df)
    games = set_fcs_opponent_ratings(games)
    games = add_playoff_points(games)
    games["playoffPoints"] = games["actualPlayoffPoints"]

    games = games[games['teamClassification'] == 'fbs']

    final_columns = [
        'id', 'season', 'week', 'seasonType', 'startDate', 'startTime', 'startTimeTBD', 'completed', 'team', 'opponent',
        'location', 'teamPoints', 'opponentPoints', 'result', 'MOV', 'conferenceGame', 'attendance', 'venue',
        'teamConference', 'opponentConference', 'teamClassification', 'opponentClassification', 'teamPregameElo',
        'opponentPregameElo', 'teamPostgameElo', 'opponentPostgameElo',
        'teamPregameRating', 'opponentPregameRating', 'teamPostgameRating', 'opponentPostgameRating',
        'teamRank', 'opponentRank', 'potentialPlayoffPoints', 'actualPlayoffPoints',
        'notes'
    ]

    os.makedirs("data", exist_ok=True)
    games[final_columns].to_csv("data/game_data.csv", index=False)
    rankings.to_csv("data/rankings.csv", index=False)
    print("✅ Game data saved to data/game_data.csv")


if __name__ == "__main__":
    main()
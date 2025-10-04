import pandas as pd
import numpy as np
import os

def calculate_win_streak(results):
    streak = 0
    for result in results:
        if result == 'Win':
            streak += 1
        else:
            break
    return streak

def main():
    game_data = pd.read_csv("data/game_data.csv")
    rankings_df = pd.read_csv("data/rankings.csv")

    conf_champ_games = [
        "ACC Championship", "Big 12 Championship", "Big Ten Championship", "Conference USA Championship",
        "MAC Championship", "Mountain West Championship", "Pac-12 Championship Game", "SEC Championship",
        "Sun Belt Championship", "American Athletic Championship"
    ]

    game_data['isConferenceTitleGame'] = game_data['notes'].fillna('').str.lower().apply(
        lambda note: any(phrase.lower() in note for phrase in conf_champ_games)
    )

    regular = game_data[game_data['seasonType'] == 'regular'].copy()
    regular['conferenceChampion'] = ((regular['isConferenceTitleGame']) & (regular['result'] == 'Win')).astype(int)

    regular_sorted = regular.sort_values(by=['season', 'team', 'week'], ascending=[True, True, False])
    win_streaks = (
        regular_sorted.groupby(['season', 'team'])['result']
        .apply(calculate_win_streak)
        .reset_index(name='EOSWStreak')
    )
    regular = regular.merge(win_streaks, on=['season', 'team'], how='left')

    elo_mean = regular['opponentPregameElo'].mean()
    elo_std = regular['opponentPregameElo'].std()
    regular['opponentEloNorm'] = (regular['opponentPregameElo'] - elo_mean) / elo_std

    final_playoff_ranks = rankings_df[rankings_df['poll'] == 'Playoff Committee Rankings']
    last_playoff_weeks = final_playoff_ranks.groupby('season')['week'].max().reset_index().rename(columns={'week': 'finalWeek'})
    final_playoff = final_playoff_ranks.merge(last_playoff_weeks, left_on=['season', 'week'], right_on=['season', 'finalWeek'])
    final_playoff['madeTop12'] = (final_playoff['rank'] <= 12).astype(int)
    final_playoff = final_playoff.rename(columns={'rank': 'endOfSeasonRank'})

    season_data = regular.groupby(['season', 'team']).agg(
        gamesPlayed=('result', 'count'),
        wins=('result', lambda x: (x == 'Win').sum()),
        losses=('result', lambda x: (x == 'Loss').sum()),
        roadWins=('location', lambda x: ((x == 'Away') & (regular.loc[x.index, 'result'] == 'Win')).sum()),
        top10Wins=('opponentRank', lambda x: ((x <= 10) & x.notna()).sum()),
        rankedWins=('opponentRank', lambda x: ((x <= 25) & x.notna()).sum()),
        conferenceChampion=('conferenceChampion', 'max'),
        avgMOV=('MOV', 'mean'),
        stdMOV=('MOV', 'std'),
        totalPoints=('teamPoints', 'sum'),
        totalPointsAllowed=('opponentPoints', 'sum'),
        avgOpponentRatingNorm=('opponentEloNorm', 'mean'),
        EOSWStreak=('EOSWStreak', 'max'),
        bestWin=('opponentEloNorm', lambda x: x[regular.loc[x.index, 'result'] == 'Win'].max()),
        worstLoss=('opponentEloNorm', lambda x: x[regular.loc[x.index, 'result'] == 'Loss'].min()),
        totalPlayoffPoints=('playoffPoints', 'sum')
    ).reset_index()

    season_data = season_data.merge(
        final_playoff[['season', 'team', 'endOfSeasonRank', 'madeTop12']],
        on=['season', 'team'],
        how='left'
    )

    season_data['madeTop12'] = season_data['madeTop12'].fillna(0).astype(int)
    os.makedirs("data", exist_ok=True)
    season_data.to_csv("data/season_data.csv", index=False)
    print("Season-level data saved to data/season_data.csv")

if __name__ == "__main__":
    main()
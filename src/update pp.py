import pandas as pd

# === Load the data ===
df = pd.read_csv("data/game_data.csv")

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

# === Save the updated file ===
final_columns = [
        'id', 'season', 'week', 'seasonType', 'startDate', 'startTime', 'startTimeTBD', 'completed', 'team', 'opponent',
        'location', 'teamPoints', 'opponentPoints', 'result', 'MOV', 'conferenceGame', 'attendance', 'venue',
        'teamConference', 'opponentConference', 'teamClassification', 'opponentClassification', 'teamPregameElo',
        'opponentPregameElo', 'teamPostgameElo', 'opponentPostgameElo',
        'teamPregameRating', 'opponentPregameRating', 'teamPostgameRating', 'opponentPostgameRating',
        'teamRank', 'opponentRank', 'potentialPlayoffPoints', 'actualPlayoffPoints',
        'notes'
    ]

df[final_columns].to_csv("data/game_data.csv", index=False)
print("✅ Playoff points updated in game_data.csv")
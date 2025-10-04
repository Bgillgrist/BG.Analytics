import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load data
game_data = pd.read_csv("data/game_data.csv")
game_data["startDate"] = pd.to_datetime(game_data["startDate"])
game_data["season"] = pd.to_datetime(game_data["startDate"]).dt.year

# Sort for rolling stats
game_data = game_data.sort_values(["team", "startDate"])

# Create rolling features
for col in ["teamPoints", "opponentPoints", "MOV"]:
    game_data[f"avg_{col}_last5"] = (
        game_data.groupby("team")[col]
        .shift(1)
        .rolling(window=5, min_periods=5)
        .mean()
        .reset_index(drop=True)
    )

# Drop rows without enough rolling history
game_data = game_data[game_data["avg_teamPoints_last5"].notna()].copy()

# Create features
team_dummies = pd.get_dummies(game_data["team"], prefix="team", drop_first=True)
opp_dummies = pd.get_dummies(game_data["opponent"], prefix="opp", drop_first=True)
location_dummies = pd.get_dummies(game_data["location"], prefix="loc", drop_first=True)
game_data["isFCS"] = (
    game_data.get("opponentClassification", "").astype(str).str.strip().str.lower().eq("fcs").astype(int)
)

# Combine all features
X_full = pd.concat([
    game_data[[
        "teamPregameRating", "opponentPregameRating", "isFCS",
        "avg_teamPoints_last5", "avg_opponentPoints_last5", "avg_MOV_last5"
    ]],
    location_dummies, team_dummies, opp_dummies
], axis=1)

y_full = game_data["MOV"]

# Drop any NaNs
X_full = X_full.loc[y_full.notna()]
y_full = y_full.loc[y_full.notna()]
X_full = X_full.dropna()
y_full = y_full.loc[X_full.index]

# Split train (before 2024) and test (2024 only)
train_idx = game_data.loc[X_full.index, "season"] < 2024
test_idx = game_data.loc[X_full.index, "season"] == 2024

X_train, y_train = X_full[train_idx], y_full[train_idx]
X_test, y_test = X_full[test_idx], y_full[test_idx]

# Train XGBoost model with basic tuning
model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predict and evaluate
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_mae = mean_absolute_error(y_train, train_preds)
test_mae = mean_absolute_error(y_test, test_preds)

print(f"Train MAE: {train_mae:.2f}")
print(f"Test MAE (2024 games): {test_mae:.2f}")

# Feature importance
importances = model.feature_importances_
features = X_train.columns
sorted_idx = importances.argsort()[::-1]
for i in sorted_idx[:15]:
    print(f"{features[i]}: {importances[i]:.4f}")

# Plot actual vs predicted on test set
plt.figure(figsize=(8,6))
plt.scatter(y_test, test_preds, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel("Actual MOV")
plt.ylabel("Predicted MOV")
plt.title("Test Set MOV Prediction (2024) with XGBoost")
plt.grid(True)
plt.tight_layout()
plt.show()
import nfl_data_py as nfl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import os
from datetime import datetime

def get_and_prepare_player_data():
    data = pd.read_csv('2024-09-12.csv')
        
    # Create df_yards: Extracting game date and yards gained for completed passes
    df_yards = data[data['complete_pass'] == 1][['game_id', 'game_date', 'passer_player_id', 'yards_gained', 'passer_name']].copy()
    print(df_yards.head())
    
    # Create df_receptions: Counting the number of receptions (complete passes) per game by date
    df_receptions = data[data['complete_pass'] == 1].groupby(['game_id','game_date', 'passer_player_id', 'passer_name']).size().reset_index(name='receptions')
    
    return df_yards, df_receptions

df_yards, df_receptions = get_and_prepare_player_data()

num_games = df_receptions['game_id'].nunique()  # Total games for sample size (probably career games)
window = num_games  # number of games we want to look at for the player (sets moving average window)

print(df_receptions.head())

# Initialize the results list
# Assuming df_receptions is already loaded
df_yards, df_receptions = get_and_prepare_player_data()

# Ensure that the games are sorted by 'passer_player_id' and 'game_date'
df_receptions = df_receptions.sort_values(by=['passer_player_id', 'game_date'])

# Initialize the results list
results = []

# Define the ranges for window sizes and alpha (decay) values
window_sizes = range(36,44, 1)  # Window sizes from 3 to 50 in steps of 4
alpha_values = np.arange(0.95, 1, 0.025)  # Alpha (decay) values from 0.5 to 1.0 in steps of 0.05
regression_factors = np.arange(7, 9, .01)
df_receptions = df_receptions.sort_values(by=['passer_name', 'game_date'])

def weighted_moving_average(x, decay):
    # Create weights that decay by 0.95 for each previous game
    weights = np.array([decay **i for i in range(len(x))])
    # Compute the weighted average (reversed weights to give more importance to recent games)
    return np.dot(x, weights[::-1]) / weights.sum()

# Loop over each combination of window size and alpha value
for window_size in window_sizes:
    for alpha in alpha_values:
        for regression_factor in regression_factors:
            # Create a copy of df_receptions to avoid modifying the original data
            df_copy = df_receptions.copy()

            # Group by player and apply exponential moving average using alpha (decay) and window size
            # Apply the weighted moving average for the last 10 games, excluding the current game (using shift)
            df_copy['wtd_rec'] = df_receptions.groupby('passer_name')['receptions']\
                .transform(lambda x: x.shift(1).rolling(window_size, min_periods=1)\
                .apply(weighted_moving_average, raw=True, args=(alpha,)))

            df_copy['regression_amount'] = 25
            df_copy['regression_games'] = window_size / regression_factor
            df_copy['wtd_rec'] = df_copy['wtd_rec'] * (1 - (df_copy['regression_games'] / window_size)) + (df_copy['regression_amount'] * (df_copy['regression_games'] / window_size))

            # Filter df_copy to only include game dates after 05/01/2021
            df_copy = df_copy[df_copy['game_date'] > '2021-05-01']

            # Calculate the RMSE between the rolling average and the actual receptions
            rmse = np.sqrt(np.mean((df_copy['receptions'] - df_copy['wtd_rec'])**2))

            # Store the results for each combination of window size and decay
            results.append({
                'window_size': window_size,
                'alpha': alpha,
                'rmse': rmse,
                'regression_factor' : regression_factor
            })

            print(f"Window Size: {window_size}, Alpha (Decay): {alpha}, RMSE: {rmse:.4f}, regression_factor: {regression_factor}")

# Sort results by RMSE and print the best combination
sorted_results = sorted(results, key=lambda x: x['rmse'])
best_result = sorted_results[0]
print(f"Best Window Size: {best_result['window_size']}, Best Alpha: {best_result['alpha']}, RMSE: {best_result['rmse']:.4f}, Best Regression Factor: {best_result['regression_factor']}")

df_receptions['wtd_rec'] = df_receptions.groupby('passer_name')['receptions'].transform(lambda x: x.shift(1).rolling(best_result['window_size'], min_periods=1).apply(weighted_moving_average, raw=True, args=(best_result['alpha'],)))
df_receptions['regression_amount'] = 3.05
df_receptions['regression_games'] = best_result['window_size'] / best_result['regression_factor']
df_receptions['wtd_rec'] = df_receptions['wtd_rec'] * (1 - (df_receptions['regression_games'] / best_result['window_size'])) + (df_receptions['regression_amount'] * (df_receptions['regression_games'] / best_result['window_size']))
df_receptions.to_csv('weighted_moving_average.csv', index=False)
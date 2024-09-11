import os
from datetime import datetime

import nfl_data_py as nfl
import pandas as pd


# Cache Data
def cache_data(seasons=[2021, 2022, 2023]):
    today = datetime.today().strftime('%Y-%m-%d')
    filename = f"{today}.csv"
    if os.path.exists(filename):
        return filename
    
    data = nfl.import_pbp_data(seasons)
    data.drop(columns=['name'], inplace=True)

    players = nfl.import_ids()
    players = players.dropna(subset=['gsis_id']).drop_duplicates(subset=['gsis_id'])
    data = data.merge(players, how='inner', left_on='receiver_player_id', right_on='gsis_id')
    data.to_csv(filename)
    return filename

# Prepare Player Data
def get_and_prepare_player_data(player_name, seasons=[2021, 2022, 2023]):
    # Cache data if not already cached
    filename = cache_data(seasons)
    
    # Specify only the columns you care about
    columns_to_use = ['receiver_player_id', 'game_id', 'game_date', 'complete_pass', 
                      'yards_gained', 'season_type', 'name']
    
    # Read only the necessary columns from the CSV file
    data = pd.read_csv(filename, usecols=columns_to_use, low_memory=False)
    
    # Filter by player name and copy to avoid SettingWithCopyWarning
    player_data = data[data['name'] == player_name].copy()
    
    # Convert 'game_date' to datetime format
    player_data['game_date'] = pd.to_datetime(player_data['game_date'], errors='coerce')
    
    # Filter out invalid dates
    player_data = player_data[player_data['game_date'].notna()]

    # Extract yards data
    df_yards = player_data[player_data['complete_pass'] == 1][['game_id', 'game_date', 'yards_gained']].copy()
    
    # Extract and aggregate reception data
    df_receptions = player_data[(player_data['complete_pass'] == 1) & (player_data['season_type'] == 'REG')].groupby(['game_date']).size().reset_index(name='receptions')
    
    # Sort df_receptions by game_date and add player_name column
    df_receptions = df_receptions.sort_values(by=['game_date']).assign(name=player_name)

    return df_yards, df_receptions

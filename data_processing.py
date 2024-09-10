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
def get_and_prepare_player_data(player_name, seasons = [2021, 2022, 2023]):
    filename = cache_data(seasons)
    data = pd.read_csv(filename, low_memory=False)
    
    player_data = data[data['name'] == player_name].copy()  # Create a copy to avoid the warning
    
    # Convert 'game_date' to datetime format, errors='coerce' will replace invalid dates with NaT
    player_data.loc[:, 'game_date'] = pd.to_datetime(player_data['game_date'], errors='coerce')
    
    # Filter out rows where 'game_date' is NaT (invalid dates)
    player_data = player_data[player_data['game_date'].notna()]

    # Extract yards data
    df_yards = player_data[player_data['complete_pass'] == 1][['game_id', 'game_date', 'yards_gained']].copy()
    
    # Use the specified line for df_receptions
    df_receptions = player_data[(player_data['complete_pass'] == 1) & (player_data['season_type'] == 'REG')].groupby(['game_date']).size().reset_index(name='receptions')
    
    # Sort df_receptions by game_date and add player_name to every column in one line
    df_receptions = df_receptions.sort_values(by=['game_date']).assign(name=player_name) #MAKE SURE THIS CAN BE DONE AT ONCE

    return df_yards, df_receptions

#get_and_prepare_player_data('Davante Adams')
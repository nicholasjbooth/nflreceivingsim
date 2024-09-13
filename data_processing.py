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
    
    # Merge receiver and passer separately to keep their names distinct
    data = data.merge(players[['gsis_id', 'name']], how='left', left_on='receiver_player_id', right_on='gsis_id').rename(columns={'name': 'receiver_name'})
    data = data.merge(players[['gsis_id', 'name']], how='left', left_on='passer_player_id', right_on='gsis_id').rename(columns={'name': 'passer_name'})
    
    # Drop unnecessary gsis_id columns
    data.drop(columns=['gsis_id_x', 'gsis_id_y'], inplace=True)
    
    # Save the data to a CSV file
    data.to_csv(filename, index=False)
    return filename

# Prepare Player Data Based on Position
def get_and_prepare_player_data(player_name, seasons=[2021, 2022, 2023]):
    # Cache data if not already cached
    filename = cache_data(seasons)

    # Load player information to find their position
    players = nfl.import_ids()
    player_info = players[players['name'] == player_name].iloc[0]  # Get player's info
    player_position = player_info['position']  # Extract the player's position

    # Specify only the columns you care about
    columns_to_use = ['receiver_player_id', 'passer_player_id', 'game_id', 'game_date', 'complete_pass', 
                      'yards_gained', 'season_type', 'receiver_name', 'passer_name']

    # Read only the necessary columns from the cached CSV file
    data = pd.read_csv(filename, usecols=columns_to_use, low_memory=False)

    # Convert 'game_date' to datetime format
    data['game_date'] = pd.to_datetime(data['game_date'], errors='coerce')

    # Filter out invalid dates
    data = data[data['game_date'].notna()]

    # Handle QB or non-QB positions dynamically based on the player's position
    if player_position == 'QB':
        # For QB: return completions and yards data frames
        qb_data = data[data['passer_name'] == player_name].copy()
        
        # Remove any rows where the QB is listed as a receiver (anomaly receptions)
        qb_data = qb_data[qb_data['receiver_name'] != player_name]

        # Create completions data frame for the QB
        df_completions = qb_data[(qb_data['complete_pass'] == 1) & (qb_data['season_type'] == 'REG')] \
            .groupby(['game_date', 'passer_name']).size().reset_index(name='completions')

        # Create yards data frame for the QB with receiver and passer names
        df_yards = qb_data[qb_data['complete_pass'] == 1][['game_id', 'game_date', 'yards_gained', 'receiver_name', 'passer_name']].copy()

        # All passer names will be the QB's name
        df_yards['passer_name'] = player_name

        return df_yards, df_completions, player_position

    else:
        # For non-QB: return receptions and yards data frames
        receiver_data = data[data['receiver_name'] == player_name].copy()

        # Create receptions data frame for the receiver
        df_receptions = receiver_data[(receiver_data['complete_pass'] == 1) & (receiver_data['season_type'] == 'REG')] \
            .groupby(['game_date', 'receiver_name']).size().reset_index(name='receptions')

        # Create yards data frame for the receiver with both receiver and passer names
        df_yards = receiver_data[receiver_data['complete_pass'] == 1][['game_id', 'game_date', 'yards_gained', 'receiver_name', 'passer_name']].copy()

        return df_yards, df_receptions, player_position

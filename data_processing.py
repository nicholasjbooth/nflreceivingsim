import os
from datetime import datetime
import nfl_data_py as nfl
import pandas as pd

# Function to load data in chunks and concatenate it
def load_data_in_chunks(file_path, chunksize=10000):
    df_list = []
    for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
        df_list.append(chunk)
    return pd.concat(df_list, axis=0)

# Cache Data
def cache_data(seasons=[2022, 2023, 2024]):
    today = datetime.today().strftime('%Y-%m-%d')
    filename = f"{today}.csv"
    if os.path.exists(filename):
        return filename

    # Fetch and process play-by-play data
    data = nfl.import_pbp_data(seasons)
    data.drop(columns=['name'], inplace=True)

    # Fetch player IDs and clean up
    players = nfl.import_ids()
    players = players.dropna(subset=['gsis_id']).drop_duplicates(subset=['gsis_id'])

    # Merge receiver and passer names separately
    data = data.merge(
        players[['gsis_id', 'name']],
        how='left',
        left_on='receiver_player_id',
        right_on='gsis_id').rename(columns={'name': 'receiver_name'})
    data = data.merge(
        players[['gsis_id', 'name']],
        how='left',
        left_on='passer_player_id',
        right_on='gsis_id').rename(columns={'name': 'passer_name'})

    # Drop unnecessary gsis_id columns
    data.drop(columns=['gsis_id_x', 'gsis_id_y'], inplace=True)

    # Save the processed data to CSV in chunks
    data.to_csv(filename, index=False)
    return filename

# Prepare Player Data Based on Position
def get_and_prepare_player_data(player_name, seasons=[2022, 2023, 2024]):
    # Cache data if not already cached
    filename = cache_data(seasons)

    # Load player information to find their position
    players = nfl.import_ids()
    player_info = players[players['name'] == player_name].iloc[0]  # Get player's info
    player_position = player_info['position']  # Extract the player's position

    # Specify only the columns you care about
    columns_to_use = [
        'receiver_player_id', 'passer_player_id', 'game_id', 'game_date',
        'complete_pass', 'yards_gained', 'season_type', 'receiver_name',
        'passer_name'
    ]

    # Load data in chunks using the chunked function
    data = load_data_in_chunks(filename)

    # Convert 'game_date' to datetime format
    data['game_date'] = pd.to_datetime(data['game_date'], errors='coerce')

    # Filter out invalid dates
    data = data[data['game_date'].notna()]

    # Handle QB or non-QB positions dynamically based on the player's position
    if player_position == 'QB':
        # For QB: return receptions and yards data frames
        qb_data = data[data['passer_name'] == player_name].copy()

        # Create receptions data frame for the QB
        df_receptions = qb_data[(qb_data['complete_pass'] == 1) & (qb_data['season_type'] == 'REG')] \
            .groupby(['game_date', 'passer_name']).size().reset_index(name='receptions')

        # Create yards data frame for the QB with receiver and passer names
        df_yards = qb_data[qb_data['complete_pass'] == 1][[
            'game_id', 'game_date', 'yards_gained', 'receiver_name',
            'passer_name'
        ]].copy()

        # All passer names will be the QB's name
        df_yards['passer_name'] = player_name

        # Save the yards dataframe to CSV for debugging or future use
        df_yards.to_csv('qb_yards.csv', index=False)

        return df_yards, df_receptions, player_position

    else:
        # For non-QB: return receptions and yards data frames
        receiver_data = data[data['receiver_name'] == player_name].copy()

        # Create receptions data frame for the receiver
        df_receptions = receiver_data[(receiver_data['complete_pass'] == 1) & (receiver_data['season_type'] == 'REG')] \
            .groupby(['game_date', 'receiver_name']).size().reset_index(name='receptions')

        # Create yards data frame for the receiver with both receiver and passer names
        df_yards = receiver_data[receiver_data['complete_pass'] == 1][[
            'game_id', 'game_date', 'yards_gained', 'receiver_name',
            'passer_name'
        ]].copy()

        return df_yards, df_receptions, player_position

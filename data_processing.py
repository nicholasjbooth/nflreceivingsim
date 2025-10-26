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
def cache_data(seasons=[2022, 2023, 2024, 2025]):
    today = datetime.today().strftime('%Y-%m-%d')
    filename = f"{today}.csv"
    if os.path.exists(filename):
        # Check if player_names.csv exists, if not create it
        if not os.path.exists('player_names.csv'):
            data = pd.read_csv(filename, low_memory=False)
            receiver_names = data['receiver_name'].dropna().unique().tolist()
            passer_names = data['passer_name'].dropna().unique().tolist()
            unique_player_names = list(set(receiver_names + passer_names))
            player_names_df = pd.DataFrame(unique_player_names, columns=['player_name'])
            player_names_df.to_csv('player_names.csv', index=False)
        return filename

    # Fetch and process play-by-play data
    data = nfl.import_pbp_data(seasons)

    # Fetch player IDs and clean up
    players = nfl.import_ids()
    players = players.dropna(subset=['gsis_id']).drop_duplicates(subset=['gsis_id'])
    
    # Rename the name column in players to avoid conflicts
    players = players.rename(columns={'name': 'player_name'})

    # Merge receiver names
    data = data.merge(
        players[['gsis_id', 'player_name']],
        how='left',
        left_on='receiver_player_id',
        right_on='gsis_id')
    data = data.rename(columns={'player_name': 'receiver_name'})
    data = data.drop(columns=['gsis_id'])
    
    # Merge passer names
    data = data.merge(
        players[['gsis_id', 'player_name']],
        how='left',
        left_on='passer_player_id',
        right_on='gsis_id')
    data = data.rename(columns={'player_name': 'passer_name'})
    data = data.drop(columns=['gsis_id'])

    # Save the processed data to CSV in chunks
    data.to_csv(filename, index=False)

    # Extract unique player names from both receiver and passer columns
    receiver_names = data['receiver_name'].dropna().unique().tolist()
    passer_names = data['passer_name'].dropna().unique().tolist()

    # Combine the lists and remove duplicates
    unique_player_names = list(set(receiver_names + passer_names))

    # Save the unique player names to player_names.csv
    player_names_df = pd.DataFrame(unique_player_names, columns=['player_name'])
    player_names_df.to_csv('player_names.csv', index=False)

    return filename

# Prepare Player Data Based on Position
def get_and_prepare_player_data(player_name, seasons=[2022, 2023, 2024, 2025]):
    # Cache data if not already cached
    filename = cache_data(seasons)

    # Load player information to find their position
    players = nfl.import_ids()
    player_info = players[players['name'] == player_name].iloc[0]
    player_position = player_info['position']

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

        # Create yards data frame for the QB with additional game info
        df_yards = qb_data[qb_data['complete_pass'] == 1][[
            'game_id', 'game_date', 'week', 'yards_gained', 'receiver_name',
            'passer_name', 'home_team', 'away_team', 'posteam', 'home_score', 'away_score'
        ]].copy()

        # All passer names will be the QB's name
        df_yards['passer_name'] = player_name

        return df_yards, df_receptions, player_position

    else:
        # For non-QB: return receptions and yards data frames
        receiver_data = data[data['receiver_name'] == player_name].copy()

        # Create receptions data frame for the receiver
        df_receptions = receiver_data[(receiver_data['complete_pass'] == 1) & (receiver_data['season_type'] == 'REG')] \
            .groupby(['game_date', 'receiver_name']).size().reset_index(name='receptions')

        # Create yards data frame for the receiver with additional game info
        df_yards = receiver_data[receiver_data['complete_pass'] == 1][[
            'game_id', 'game_date', 'week', 'yards_gained', 'receiver_name',
            'passer_name', 'home_team', 'away_team', 'posteam', 'home_score', 'away_score'
        ]].copy()

        return df_yards, df_receptions, player_position
import logging
import os

from flask import Flask, jsonify, render_template, request, session
from sklearn.mixture import GaussianMixture

from calculations import calculate_thresholds, weighted_moving_average
from data_processing import get_and_prepare_player_data
from simulation import simulate_games

# Create and configure the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get(
    'SECRET_KEY', 'default_secret_key')  # Added default value for testing

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Constants
num_sims = 10000
window = 7
alpha = 0.915
regression_amount = 3.05
regression_games = window / 5

receptions_thresholds = [2, 3, 4, 5, 6, 7, 8, 9]
yards_thresholds = [25, 40, 50, 60, 70, 80, 90, 100, 110, 125, 150]
longest_reception_thresholds = [10, 20, 30, 40, 50]


# Error handling
@app.errorhandler(Exception)
def handle_exception(e):
    logging.exception("An error occurred")
    return jsonify(error=str(e)), 500


@app.route('/')
def index():
    # Get session values if available
    player_name = session.get('player_name', '')
    yard_threshold = session.get('yard_threshold', '')
    receptions_threshold = session.get('receptions_threshold', '')
    current_yards = session.get('current_yards', '')
    current_receptions = session.get('current_receptions', '')
    time_remaining = session.get('time_remaining', '')

    # Pass the session values to the index.html template
    return render_template(
        'index.html',
        player_name=player_name,
        yard_threshold=yard_threshold,
        receptions_threshold=receptions_threshold,
        current_yards=current_yards,
        current_receptions=current_receptions,
        time_remaining=time_remaining,
    )


@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        # Get form inputs and store them in local variables
        player_name = request.form.get('player_name')
        yard_threshold = request.form.get('yard_threshold', None)
        receptions_threshold = request.form.get('receptions_threshold', None)
        current_yards = int(request.form.get('current_yards') or 0)
        current_receptions = int(request.form.get('current_receptions') or 0)
        time_remaining = int(request.form.get('time_remaining') or 60)

        # Store the form inputs in the session for future use
        session['player_name'] = player_name
        session['yard_threshold'] = yard_threshold
        session['receptions_threshold'] = receptions_threshold
        session['current_yards'] = current_yards
        session['current_receptions'] = current_receptions
        session['time_remaining'] = time_remaining

        # Get and prepare player data
        df_yards, df_receptions = get_and_prepare_player_data(player_name)
        yards_per_reception = df_yards['yards_gained']

        df_receptions['wtd_rec'] = df_receptions.groupby(
            'name')['receptions'].transform(
                lambda x: x.shift(0).rolling(window, min_periods=1).apply(
                    weighted_moving_average, raw=True, args=(
                        alpha, )))  #maybe put this into data processing

        df_receptions['wtd_rec'] = df_receptions['wtd_rec'] * (
            1 - (regression_games / window)) + (regression_amount *
                                                (regression_games / window))
        average_receptions = df_receptions['wtd_rec'].iloc[
            -1]  #model input variable
        adj_average = average_receptions * (
            time_remaining / 60)  # Adjusted for remaining time remaining

        gmm = GaussianMixture(n_components=3)
        gmm.fit(yards_per_reception.values.reshape(-1, 1))

        games_sim_results = simulate_games(adj_average, gmm, num_sims)
        median_yards = round(games_sim_results['Simulated_Yards'].median(),
                             2)  # Rounded median yards

        median_longest_reception = round(
            games_sim_results['Longest_Reception'].median(),
            2)  # Median of longest receptions

        # Round adjusted average to 2 decimals
        adj_average = round(adj_average, 2)

        yard_threshold = median_yards if not yard_threshold else float(
            yard_threshold)

        if not receptions_threshold:
            receptions_threshold = df_receptions['receptions'].median(
            )  #uses historical data as num recs
        else:
            receptions_threshold = float(receptions_threshold)

        # Calculate the thresholds
        threshold_results = calculate_thresholds(games_sim_results,
                                                 yard_threshold,
                                                 receptions_threshold,
                                                 receptions_thresholds,
                                                 yards_thresholds,
                                                 longest_reception_thresholds)

        def format_odds(odds_value):
            return f"+{int(odds_value)}" if odds_value >= 100 else int(
                odds_value)

        # Helper function to format percentages
        def format_percentage(percent_value):
            return f"+{percent_value:.2f}%" if percent_value >= 100 else f"{percent_value:.2f}%"

        # Apply formatting to all results
        def format_results(results_list):
            formatted_list = []
            for result in results_list:
                percent_str, odds_str = result.split("% ")
                percent_value = float(percent_str.split(": ")[1].strip("%"))
                odds_value = int(odds_str)
                formatted_list.append(
                    f"{percent_str.split(': ')[0]}:{format_percentage(percent_value)} {format_odds(odds_value)}"
                )
            return formatted_list

        # Format each category
        alt_recs = format_results(threshold_results['alt_recs'])
        alt_yards = format_results(threshold_results['alt_yards'])
        uyards_orecs = format_results(threshold_results['uyards_orecs'])
        urecs_oyards = format_results(threshold_results['urecs_oyards'])
        alt_longest_recs = format_results(
            threshold_results['alt_longest_recs'])

        return render_template(
            'results.html',
            player_name=player_name,
            median_yards=median_yards,
            adj_average=adj_average,
            median_longest_reception=median_longest_reception,
            alt_recs=alt_recs,
            alt_yards=alt_yards,
            uyards_orecs=uyards_orecs,
            urecs_oyards=urecs_oyards,
            alt_longest_recs=alt_longest_recs)

    except Exception as e:
        logging.exception("An error occurred during simulation")
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

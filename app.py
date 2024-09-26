import logging
import os
import time
import nfl_data_py as nfl
import pandas as pd
from flask import Flask, jsonify, render_template, request, session
from sklearn.mixture import GaussianMixture

from calculations import calculate_thresholds, weighted_moving_average
from data_processing import get_and_prepare_player_data, cache_data
from simulation import simulate_games
from betting_tools import calculate_bet_size_and_effective_odds, i2a, a2d, kelly_criterion

# Create and configure the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get(
    'SECRET_KEY', 'default_secret_key')  # Added default value for testing

# Configure logging
logging.basicConfig(level=logging.DEBUG)


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
    lower_bound = session.get('lower_bound', '')
    upper_bound = session.get('upper_bound', '')
    lower_bound_odds = session.get('lower_bound_odds', '')
    upper_bound_odds = session.get('upper_bound_odds', '')
    lower_bound_stake = session.get('lower_bound_stake', '')
    upper_bound_stake = session.get('upper_bound_stake', '')
    ytooo = session.get('ytooo', '')  #yard threshold over odds offered
    ytuoo = session.get('ytuoo', '')  #yard threshold under odds offered

    #receiver_names = pd.read_csv(cache_data())['receiver_name'].unique().tolist()
    receiver_names = pd.read_csv(cache_data(), dtype=str, low_memory=False)["receiver_name"].unique().tolist()

    # Logging session values for debugging
    logging.debug(
        f"Session values - Player: {player_name}, Yard Threshold: {yard_threshold}, "
        f"Receptions Threshold: {receptions_threshold}, Current Yards: {current_yards}, "
        f"Current Receptions: {current_receptions}, Time Remaining: {time_remaining}"
    )

    # Pass the session values to the index.html template
    return render_template('index.html',
                           player_name=player_name,
                           yard_threshold=yard_threshold,
                           receptions_threshold=receptions_threshold,
                           current_yards=current_yards,
                           current_receptions=current_receptions,
                           time_remaining=time_remaining,
                           lower_bound=lower_bound,
                           upper_bound=upper_bound,
                           lower_bound_odds=lower_bound_odds,
                           upper_bound_odds=upper_bound_odds,
                           lower_bound_stake=lower_bound_stake,
                           upper_bound_stake=upper_bound_stake,
                           ytooo=ytooo,
                           ytuoo=ytuoo,
                           receiver_names=receiver_names)


@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        # Start timing
        start_time = time.time()
        logging.debug("Simulation started.")

        # Get form inputs and store them in local variables
        player_name = request.form.get('player_name')
        yard_threshold = request.form.get('yard_threshold', None)
        receptions_threshold = request.form.get('receptions_threshold', None)
        current_yards = int(request.form.get('current_yards') or 0)
        current_receptions = int(request.form.get('current_receptions') or 0)
        time_remaining = int(request.form.get('time_remaining') or 60)
        logging.debug(
            f"Form inputs - Player: {player_name}, Yard Threshold: {yard_threshold}, "
            f"Receptions Threshold: {receptions_threshold}, Current Yards: {current_yards}, "
            f"Current Receptions: {current_receptions}, Time Remaining: {time_remaining}"
        )

        # Fetch and convert lower/upper bounds and odds
        lower_bound = float(request.form.get('lower_bound') or 55)
        upper_bound = float(request.form.get('upper_bound') or 75)
        lower_bound_odds = int(request.form.get('lower_bound_odds') or -110)
        upper_bound_odds = int(request.form.get('upper_bound_odds') or -110)
        lower_bound_stake = float(request.form.get('lower_bound_stake') or 0.0)
        upper_bound_stake = float(request.form.get('upper_bound_stake') or 0.0)

        ytooo = int(request.form.get('ytooo') or -110)
        ytuoo = int(request.form.get('ytuoo') or -110)

        logging.debug(
            f"Bounds and Odds - Lower Bound: {lower_bound}, Upper Bound: {upper_bound}, "
            f"Lower Bound Odds: {lower_bound_odds}, Upper Bound Odds: {upper_bound_odds}"
        )

        # Cache session values
        session['player_name'] = player_name
        session['yard_threshold'] = yard_threshold
        session['receptions_threshold'] = receptions_threshold
        session['current_yards'] = current_yards
        session['current_receptions'] = current_receptions
        session[
            'time_remaining'] = None if time_remaining == 60 else time_remaining
        session['lower_bound'] = lower_bound
        session['upper_bound'] = upper_bound
        session['lower_bound_odds'] = lower_bound_odds
        session['upper_bound_odds'] = upper_bound_odds
        session['lower_bound_stake'] = lower_bound_stake
        session['upper_bound_stake'] = upper_bound_stake
        session['ytooo'] = ytooo
        session['ytuoo'] = ytuoo

        data_start = time.time()
        logging.debug("Session values updated.")
        df_yards, df_receptions, player_position = get_and_prepare_player_data(
            player_name)

        # Time player data preparation
        bankroll = 1000
        if player_position == 'QB':
            # Constants for qbs
            num_sims = 15000
            window = 39
            alpha = 0.99
            regression_amount = 18.65
            regression_games = window / 4.5

            receptions_thresholds = [10, 15, 20, 25, 30, 35, 40, 45, 50]
            yards_thresholds = [
                150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400
            ]
            longest_reception_thresholds = [30, 40, 50, 60, 70]
        else:
            num_sims = 15000
            window = 7
            alpha = 0.915
            regression_amount = 3.05
            regression_games = window / 5
            receptions_thresholds = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            yards_thresholds = [
                25, 40, 50, 60, 70, 80, 90, 100, 110, 125, 150, 200
            ]
            longest_reception_thresholds = [10, 20, 30, 40, 50, 60]

        logging.debug(
            f"Data preparation took {time.time() - data_start:.2f} seconds")

        # Apply calculations based on the player's position
        if player_position == 'QB':
            # For QBs, use passer_name and receptions
            df_receptions['wtd_rec'] = df_receptions.groupby(
                'passer_name')['receptions'].transform(
                    lambda x: x.shift(0).rolling(window, min_periods=1).apply(
                        weighted_moving_average, raw=True, args=(alpha, )))
        else:
            # For non-QBs, use receiver_name and receptions
            df_receptions['wtd_rec'] = df_receptions.groupby(
                'receiver_name')['receptions'].transform(
                    lambda x: x.shift(0).rolling(window, min_periods=1).apply(
                        weighted_moving_average, raw=True, args=(alpha, )))

        logging.debug(
            f"Weighted moving average calculation took {time.time() - data_start:.2f} seconds"
        )

        df_receptions['wtd_rec'] = df_receptions['wtd_rec'] * (
            1 - (regression_games / window)) + (regression_amount *
                                                (regression_games / window))
        average_receptions = df_receptions['wtd_rec'].iloc[-1]
        adj_average = average_receptions * (time_remaining / 60)
        logging.debug(f"Adjusted Average: {adj_average}")

        # Time GMM fitting
        gmm_start = time.time()
        yards_per_reception = df_yards['yards_gained']
        gmm = GaussianMixture(n_components=3)
        gmm.fit(yards_per_reception.values.reshape(-1, 1))
        logging.debug(
            f"GMM fitting took {time.time() - gmm_start:.2f} seconds")

        # Time simulation
        sim_start = time.time()
        games_sim_results = simulate_games(adj_average, gmm, num_sims,
                                           current_yards, current_receptions)
        logging.debug(f"Simulation took {time.time() - sim_start:.2f} seconds")

        # Calculate medians
        median_yards = round(games_sim_results['Simulated_Yards'].median(), 2)
        median_longest_reception = round(
            games_sim_results['Longest_Reception'].median(), 2)

        adj_average = round(adj_average, 2)
        yard_threshold = median_yards if not yard_threshold else float(
            yard_threshold)
        receptions_threshold = df_receptions['receptions'].median(
        ) if not receptions_threshold else float(receptions_threshold)

        # Time threshold calculation
        threshold_start = time.time()
        threshold_results = calculate_thresholds(
            games_sim_results, yard_threshold, receptions_threshold,
            receptions_thresholds, yards_thresholds,
            longest_reception_thresholds, lower_bound, upper_bound)
        logging.debug(
            f"Threshold calculation took {time.time() - threshold_start:.2f} seconds"
        )

        # Formatting results
        def format_odds(odds_value):
            return f"+{int(odds_value)}" if odds_value >= 100 else int(
                odds_value)

        def format_percentage(percent_value):
            if percent_value is None:
                return None
            else:
                return f"+{percent_value:.2f}%" if percent_value >= 100 else f"{percent_value:.2f}%"

        def format_results(results_list):
            formatted_list = []
            for result in results_list:
                percent_str, odds_str = result.split("% ")
                percent_value = float(percent_str.split(": ")[1].strip("%"))
                odds_value = int(odds_str)
                formatted_list.append(
                    f"{percent_str.split(': ')[0]}: {format_percentage(percent_value)} {format_odds(odds_value)}"
                )
            return formatted_list

        alt_recs = format_results(threshold_results['alt_recs'])
        alt_yards = format_results(threshold_results['alt_yards'])
        uyards_orecs = format_results(threshold_results['uyards_orecs'])
        urecs_oyards = format_results(threshold_results['urecs_oyards'])
        alt_longest_recs = format_results(
            threshold_results['alt_longest_recs'])

        ytop = threshold_results[
            'ytop']  #Probability the player goes over the yard threshold
        ytup = threshold_results[
            'ytup']  #Probability the player goes under the yard threshold

        ytoood = a2d(ytooo)  #Convert American odds to decimal
        ytuood = a2d(ytuoo)  #Convert American odds to decimal

        ytok = kelly_criterion(ytoood, ytop)  #Kelly Criterion for over
        ytuk = kelly_criterion(ytuood, ytup)  #Kelly Criterion for under

        ytoba = round(ytok * .25 * bankroll,
                      2)  #Yards threshold over bet amount
        ytuba = round(ytuk * .25 * bankroll,
                      2)  #Yards threshold under bet amount

        ytoa = i2a(ytop)  #YTOP American Odds
        ytua = i2a(ytup)  #YTUP American Odds

        if lower_bound is not None and upper_bound is not None:
            percent_between = format_percentage(
                threshold_results['percent_between'])
            percent_between_american = i2a(
                (threshold_results['percent_between']) / 100)
        else:
            percent_between = None
            percent_between_american = None

        total_time = time.time() - start_time
        logging.debug(f"Total simulation took {total_time:.2f} seconds")

        if lower_bound is not None and upper_bound is not None:
            # Handle the calculation of bet size and odds based on whether upper_bound_stake has a value
            if upper_bound_stake:  # Check if upper_bound_stake is not 0
                bet_on = 'outcome_2'
                stake = upper_bound_stake
            else:
                bet_on = 'outcome_1'
                stake = lower_bound_stake

            middle_metrics = calculate_bet_size_and_effective_odds(
                lower_bound_odds, upper_bound_odds, stake, bet_on=bet_on)

            lower_bound_stake = middle_metrics['bet_size_1']
            upper_bound_stake = middle_metrics['bet_size_2']
            risk_amount = middle_metrics['risk_amount']
            effective_odds = middle_metrics['effective_odds']
            effective_american = i2a(effective_odds)
        else:
            # If no bet size values, set default or skip this part
            lower_bound_stake = None
            upper_bound_stake = None
            risk_amount = None
            effective_american = None
            effective_odds = None

        # Format for output
        effective_odds = format_percentage(effective_odds * 100)
        effective_american = format_odds(effective_american)
        percent_between_american = format_odds(percent_between_american)

        risk_amount = round(risk_amount, 2)
        lower_bound_stake = round(lower_bound_stake, 2)
        upper_bound_stake = round(upper_bound_stake, 2)

        ytop = format_percentage(ytop)
        ytup = format_percentage(ytup)

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
            alt_longest_recs=alt_longest_recs,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            lower_bound_odds=lower_bound_odds,
            upper_bound_odds=upper_bound_odds,
            lower_bound_stake=lower_bound_stake,
            upper_bound_stake=upper_bound_stake,
            percent_between=percent_between,
            effective_odds=effective_odds,
            risk_amount=risk_amount,
            percent_between_american=percent_between_american,
            effective_american=effective_american,
            ytoba=ytoba,
            ytuba=ytuba,
            ytoa=ytoa,
            ytua=ytua,
            yard_threshold=yard_threshold,
            ytooo=ytooo,
            ytuoo=ytuoo,
        )

    except Exception as e:
        logging.exception("An error occurred during simulation")
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

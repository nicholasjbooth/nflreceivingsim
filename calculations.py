import numpy as np

from betting_tools import i2a


# Calculate the weighted moving average
def weighted_moving_average(x, decay):
    weights = np.array([decay ** i for i in range(len(x))])
    return np.dot(x, weights[::-1]) / weights.sum()

#Calculate the thresholds for printing
def calculate_thresholds(games_sim_results, yard_threshold, receptions_threshold, 
                         receptions_thresholds, yards_thresholds, longest_reception_thresholds, lower_bound, upper_bound):
    # Categorized results
    alt_recs = []
    alt_yards = []
    uyards_orecs = []
    urecs_oyards = []
    alt_longest_recs = []    

    # Receptions threshold results (Alt Recs)
    for threshold in receptions_thresholds:
        percent_above = (games_sim_results['Simulated_Receptions'] >= threshold).mean() * 100
        odds = i2a(percent_above / 100)  # Convert implied probability to American odds
        alt_recs.append(f"{threshold}+ receptions: {percent_above:.2f}% {odds}")

    # Yards threshold results (Alt Yards)
    for threshold in yards_thresholds:
        percent_above = (games_sim_results['Simulated_Yards'] >= threshold).mean() * 100
        odds = i2a(percent_above / 100)  # Convert implied probability to American odds
        alt_yards.append(f"{threshold}+ yards: {percent_above:.2f}% {odds}")

    # Under user-defined yards and over various receptions thresholds (uYards ORecs)
    for threshold in receptions_thresholds:
        percent = ((games_sim_results['Simulated_Yards'] <= yard_threshold) & 
                   (games_sim_results['Simulated_Receptions'] >= threshold)).mean() * 100
        odds = i2a(percent / 100)  # Convert implied probability to American odds
        uyards_orecs.append(f"u{yard_threshold} and {threshold}+: {percent:.2f}% {odds}")

    # Under user-defined receptions and over various yard thresholds (uRecs OYards)
    for threshold in yards_thresholds:
        percent = ((games_sim_results['Simulated_Receptions'] <= receptions_threshold) & 
                   (games_sim_results['Simulated_Yards'] >= threshold)).mean() * 100
        odds = i2a(percent / 100)  # Convert implied probability to American odds
        urecs_oyards.append(f"u{receptions_threshold} and {threshold}+: {percent:.2f}% {odds}")

    # Longest reception thresholds (Alt Longest Recs)
    for threshold in longest_reception_thresholds:
        percent_above = (games_sim_results['Longest_Reception'] > threshold).mean() * 100
        odds = i2a(percent_above / 100)  # Convert implied probability to American odds
        alt_longest_recs.append(f"{threshold}+ yard reception: {percent_above:.2f}% {odds}")

    # Calculate the probability of going between 2 yard thresholds lower_bound and upper_bound
    percent_between = ((games_sim_results['Simulated_Yards'] >= lower_bound) & (games_sim_results['Simulated_Yards'] <= upper_bound)).mean() * 100

    # Return categorized results
    return {
        'alt_recs': alt_recs,
        'alt_yards': alt_yards,
        'uyards_orecs': uyards_orecs,
        'urecs_oyards': urecs_oyards,
        'alt_longest_recs': alt_longest_recs,
        'percent_between': percent_between
    }
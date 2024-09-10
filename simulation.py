import numpy as np
import pandas as pd
from flask import Flask, render_template, request, session  # Added session
from sklearn.mixture import GaussianMixture


def simulate_receptions(adj_average, num_simulations):
    return np.round(np.random.poisson(adj_average, size=num_simulations))


def simulate_yards_per_game(num_receptions, gmm):
    if num_receptions == 0:
        return 0
    return np.sum(np.round(gmm.sample(num_receptions)[0].flatten()))


def simulate_longest_reception(num_receptions, gmm):
    if num_receptions == 0:
        return 0
    return np.max(np.round(gmm.sample(num_receptions)[0].flatten()))


def simulate_games(adj_average, gmm, num_simulations):
    simulated_total_yards = []
    simulated_receptions = []
    longest_receptions = []

    for _ in range(num_simulations):
        num_receptions = simulate_receptions(adj_average, 1)[0]
        total_yards = simulate_yards_per_game(num_receptions, gmm)
        longest_reception = simulate_longest_reception(num_receptions, gmm)
        simulated_receptions.append(num_receptions)
        simulated_total_yards.append(total_yards)
        longest_receptions.append(longest_reception)

    return pd.DataFrame({
        'Simulated_Receptions': pd.Series(simulated_receptions),
        'Simulated_Yards': pd.Series(simulated_total_yards),
        'Longest_Reception': longest_receptions
    })

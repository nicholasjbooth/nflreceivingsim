import numpy as np


# American to Decimal Odds
def a2d(american_odds):
    if american_odds > 0:
        return 1 + (american_odds / 100)
    else:
        return 1 + (100 / abs(american_odds))

# American to Implied
def a2i(american_odds):
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)

# Decimal to American Odds
def d2a(decimal_odds):
    if decimal_odds < 2:
        return round(-100 / (decimal_odds - 1))
    else:
        return round((decimal_odds - 1) * 100)
    
# Decimal to Implied
def d2i(decimal_odds):
    return 1 / decimal_odds

# Implied to American
def i2a(implied_probability):
    if implied_probability == 0:
        return 0  # Return infinity for implied probability of 0
    elif implied_probability == 1:
        return 1  # Return negative infinity for implied probability of 1
    elif implied_probability > 0.5:
        return round(-100 * implied_probability / (1 - implied_probability))
    else:
        return round(((1 - implied_probability) / implied_probability) * 100)
    
    
# Implied to Decimal
def i2d(implied_probability):
    return 1 / implied_probability

#Arbitrage Calculator with effective bet
def calculate_bet_size_and_effective_odds(odds_1_american, odds_2_american, known_bet_size=None, bet_on='outcome_1'):
    odds_1_decimal = a2d(odds_1_american)
    odds_2_decimal = a2d(odds_2_american)
    
    print(f"odds_1_decimal: {odds_1_decimal}, odds_2_decimal: {odds_2_decimal}")
    
    if known_bet_size is None:
        known_bet_size = 100

    if bet_on is None:
        bet_on = 'outcome_1'

    if bet_on == 'outcome_1':
        bet_size_2 = (known_bet_size * odds_1_decimal) / odds_2_decimal
        print(f"bet_size_2: {bet_size_2}")

    elif bet_on == 'outcome_2':
        bet_size_1 = (known_bet_size * odds_2_decimal) / odds_1_decimal
        bet_size_2 = known_bet_size
        print(f"bet_size_1: {bet_size_1}, bet_size_2: {bet_size_2}")

    else:
        raise ValueError("Invalid bet_on value. Must be 'outcome_1' or 'outcome_2'")

    effective_probability = 1 - (odds_1_decimal * odds_2_decimal) / (odds_1_decimal + odds_2_decimal)
    effective_probability = max(0, min(1, effective_probability))  # Ensure effective_probability is between 0 and 1
    risk_amount = (known_bet_size + bet_size_2) * effective_probability
    effective_probability = 1 if effective_probability < 0 else effective_probability
    
    print(f"effective_probability: {effective_probability}, risk_amount: {risk_amount}")
    
    return {
        'bet_size_1': known_bet_size if bet_on == 'outcome_1' else bet_size_1,
        'bet_size_2': bet_size_2,
        'risk_amount': risk_amount,
        'effective_odds': effective_probability
    }


'''x = calculate_bet_size_and_effective_odds(-110, -130, 500, 'outcome_2')
print(x)

y = (i2d(x['effective_odds']))*x['risk_amount']
print(y)

z = i2a(x['effective_odds'])
print(z)'''
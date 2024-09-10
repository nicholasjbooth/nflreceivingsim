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


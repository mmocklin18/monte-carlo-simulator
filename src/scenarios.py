import numpy as np

def apply_scenario(mu, sigma, scenario):
    """
    Adjust mu, sigma arrays for different market scenarios
    """
    scenario = scenario.lower()

    if scenario == "bull":
        return mu + 0.05, sigma * 0.8

    if scenario == "bear":
        return mu - 0.15, sigma * 1.7

    if scenario == "volatile":
        return mu, sigma * 2.0

    #default
    return mu, sigma

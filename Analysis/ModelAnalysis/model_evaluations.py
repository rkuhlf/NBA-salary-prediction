import math
import numpy as np
import pandas as pd
from sklearn import metrics

from path_config import *
from Data.inputs import get_player_data

def get_rmse(model, inputs_test, outputs_test):
    """
    Returns the root mean square error of evaluating a model on test data.
    """
    predictions = model.predict(inputs_test)
    mse = metrics.mean_squared_error(outputs_test, predictions)

    rmse = np.sqrt(mse)

    return rmse

def get_percent_error(model, inputs_test, outputs_test):
    """
    Returns the average percent error that the model's predictions have on test data.
    Returns a percent so 100 represents a complete fraction.
    """
    predictions = model.predict(inputs_test)
    mape = metrics.mean_absolute_percentage_error(outputs_test, predictions)

    return mape * 100

def compare_prediction(model, input_columns, name):
    player_data = get_player_data()
    player_data = player_data[player_data["name"] == name]

    if len(player_data) > 1:
        raise Exception("Multiple player names")
    if len(player_data) == 0:
        raise Exception("Could not find player")

    predicted_salary = model.predict(player_data[input_columns])

    print(f"\n---{name}---")
    print(f"Predicted is {predicted_salary[0] / 1e6:.2f} million")

    actual = player_data["adjusted_career_revenue"].values[0] / 1e6
    print(f"Actual is {actual:.2f} million")

def compare_players(model, input_columns):
    players = ["LeBron James", "Steve Nash", "Dwight Howard", "Troy Murphy", "Dante Cunningham", "Stephen Curry", "Shaquille O'Neal", "Pau Gasol", "Chris Bosh", "Kobe Bryant", "Fred Roberts", "Zydrunas Ilgauskas", "James Harden", "Russell Westbrook", "Michael Jordan"]

    for player in players:
        compare_prediction(model, input_columns, player)


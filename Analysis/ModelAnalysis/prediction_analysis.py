"""
Analyze the predictions that a model makes
"""

import pandas as pd

from Data.inputs import get_player_data_for_modeling

def predict_custom(model, custom_input: pd.DataFrame):
    return model.predict(custom_input)

def highest_predictions(model, input_columns, top=10):
    players, outputs = get_player_data_for_modeling(input_columns)

    predictions = model.predict(players)

    players["prediction"] = predictions
    players["actual"] = outputs

    return players.sort_values("prediction")[-top:]
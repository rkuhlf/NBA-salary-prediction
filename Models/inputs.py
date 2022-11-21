import pandas as pd

from config import *


def get_data(input_columns):
    players_df = pd.read_csv(CLEANED_PLAYERS_PATH)

    inputs = players_df[input_columns]
    outputs = players_df["adjusted_career_revenue"]

    return inputs, outputs


import pandas as pd




def get_data(input_columns):
    players_df = pd.read_csv("cleaned_players.csv")

    inputs = players_df[input_columns]
    outputs = players_df["adjusted_career_revenue"]

    return inputs, outputs


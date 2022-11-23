import numpy as np
import pandas as pd

from path_config import *
from Data.inputs import get_player_data

def print_top_players(data: pd.DataFrame, count=10, column="career_revenue"):
    ordered = data.sort_values(column)

    for i in range(1, count + 1):
        row = ordered.iloc[[-i]]
        name = row['name'].values[0]
        value = row[column].values[0]
        print(f"{name}: {value}")

def analyze_players_overall(players: pd.DataFrame):
    """
    Prints basic overview of the players dataframe.
    """
    print(len(players))
    print(np.std(players["adjusted_career_revenue"]))


if __name__ == "__main__":
    players_df = get_player_data()

    # analyze_players_overall(players_df)
    # print_top_players(players_df, column="adjusted_career_revenue")
    print_top_players(players_df, column="career_ws")
    
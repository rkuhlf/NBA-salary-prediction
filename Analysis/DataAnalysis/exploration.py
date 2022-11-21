import numpy as np
import pandas as pd

from path_config import *
from Data.inputs import get_player_data

def analyze_players_overall(players: pd.DataFrame):
    """
    Prints basic overview of the players dataframe.
    """
    print(len(players))
    print(np.std(players["adjusted_career_revenue"]))


if __name__ == "__main__":
    players_df = get_player_data()

    analyze_players_overall(players_df)
    
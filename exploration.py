import numpy as np
import pandas as pd

from config import *

def analyze_players_overall(players: pd.DataFrame):
    print(len(players))
    print(np.std(players["adjusted_career_revenue"]))


if __name__ == "__main__":
    players_df = pd.read_csv(CLEANED_PLAYERS_PATH)

    analyze_players_overall(players_df)
    
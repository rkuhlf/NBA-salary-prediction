



import numpy as np
import pandas as pd


if __name__ == "__main__":
    players_df = pd.read_csv("cleaned_players.csv")

    print(np.std(players_df["adjusted_career_revenue"]))
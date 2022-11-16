import pandas as pd

input_columns = ["career_g normalized",
                #  "birth_year",
                 "draft_round",
                 "draft_year",
                 "height normalized",
                 "weight normalized",
                 "career_pts normalized",
                 "career_per normalized",
                 "career_ws normalized",
                 "career_ast normalized",
                 'career_fg normalized',
                 "career_fg3 normalized",
                #  "career_trb normalized",
                 'career_ft normalized',
                #  "birth_month",
                 "attended_college",
                 ]


def get_data():
    players_df = pd.read_csv("cleaned_players.csv")

    inputs = players_df[input_columns]
    outputs = players_df["adjusted_career_revenue"]

    return inputs, outputs


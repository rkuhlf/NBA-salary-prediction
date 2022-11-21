import pandas as pd
from sklearn.preprocessing import StandardScaler
import regex as re

from config import *
from helpers import na_count
scaler = StandardScaler()


def drop_na(df):
    to_drop_columns = ["career_efg", "birthplace", "highschool", "draft_team", "draft_pick"]
    for column in to_drop_columns:
        print(f"Dropping {column}")
        df.drop(labels=[column], axis=1, inplace=True)
    print()


    to_drop_rows = ["career_ast", "career_fg", "career_g", "career_per", "career_pts", "career_trb", "career_ws", "weight", "height", "birthdate", "draft_round", "draft_year"]
    for column in to_drop_rows:
        drop_count = df[column].isna().sum()
        total_count = len(df)
        print(f"Dropping {drop_count} rows out of {total_count} ({drop_count/total_count:.2f}) because missing {column}")

        df.dropna(subset=[column], inplace=True)

    print()


    to_default = {
        "career_fg3": 0,
        "career_ft": 0,
        "shoots": "right",
        "college": "N/A"
    }

    for column, default in to_default.items():
        drop_count = df[column].isna().sum()
        total_count = len(df)
        print(f"Setting to {default} {drop_count} rows out of {total_count} ({drop_count/total_count:.2f}) because missing {column}")

        df[column].fillna(default, inplace=True)
    print()


    if na_count(df) != 0:
        print(f"Total NA: {na_count}")

        for column in df.columns:
            print(f"{column} has {df[column].isna().sum()} nan values")

        raise Exception("N/A still in data set")





    def convert_height(value):
        feet, inches = value.split("-")

        return int(feet) * 12 + int(inches)

    df["height"] = df["height"].map(convert_height)

    def convert_weight(value):
        return int(value[:-2])

    df["weight"] = df["weight"].map(convert_weight)

    def convert_date(value):
        year, month, day = value.split("-")

        return int(year), int(month), int(day)

    df["birth_year"], df["birth_month"], df["birth_day"] = zip(*df["birthdate"].map(convert_date))

    to_normalize = ["career_ast", "career_fg", "career_fg3", "career_ft", "career_g", "career_per", "career_pts","career_trb", "career_ws", "career_efg", "height", "weight"]
    for column in to_normalize:
        if column in df.columns:
            df[column + " normalized"] = scaler.fit_transform(df[column].values.reshape(-1, 1))
        else:
            print(f"Skipping {column} because it has been removed")

def convert_formats(df: pd.DataFrame):
    def convert_round(value):
        try:
            num = re.search(r'\d+', str(value)).group()
        except:
            print(f"Could not convert {value}")
            return pd.NaN

        return int(num)

    df["draft_round"] = df["draft_round"].map(convert_round)

    return df

def add_categorical(df: pd.DataFrame):
    df["attended_college"] = df["college"] != "N/A"
    df["attended_college"] = df["attended_college"].apply(int)

    positions = ["Point Guard", "Shooting Guard", "Small Forward", "Power Forward", "Center", "Forward", "Guard"]

    for position in positions:
        df[position] = df["position"].map(lambda pos : int(position in pos))

    return df

def add_career_revenue(data, from_name="salary", target_name="career_revenue"):
    data[target_name] = 0
    # clean_salaries must be run before this.
    salaries = pd.read_csv(CLEANED_SALARIES_PATH)
    for index, row in data.iterrows():
        id = row["id"]
        
        data.loc[index, target_name] = int(salaries[salaries["player_id"] == id][from_name].sum())
    
    orig_len = len(data)
    data = data[data[target_name] != 0]

    print(f"Dropped {orig_len - len(data)} out of {orig_len} because their salaries were 0.")

    return data


if __name__ == "__main__":
    df = pd.read_csv(PLAYERS_PATH)

    drop_na(df)
    print(len(df))

    df = add_categorical(df)
    print(len(df))
    df = convert_formats(df)
    print(len(df))


    # # Add total revenue for nominal salary and adjusted salary
    df = add_career_revenue(df)
    print(len(df))

    df = add_career_revenue(df, "adjusted_salary", "adjusted_career_revenue")    
    print(len(df))

    df.to_csv(CLEANED_PLAYERS_PATH, index=False)
    print(len(df))

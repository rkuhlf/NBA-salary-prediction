
from matplotlib import pyplot as plt
import pandas as pd


def show_salaries_histogram(data: pd.DataFrame):
    data = data.copy()

    data["salary"] /= 1e6

    plt.hist(data["salary"], bins=50)

    plt.title("Distribution of Salaries")
    plt.xlabel("Amount (millions)")
    plt.ylabel("Number of Salaries")

    plt.show()

def show_salaries_by_team(data: pd.DataFrame):
    # TODO
    data = data.copy()

    print(data.groupby("team").mean())

def show_career_revenue_distribution(data: pd.DataFrame):
    data = data.copy()

    data["career_revenue"] /= 1e6

    plt.hist(data["career_revenue"], bins=50)

    plt.title("Distribution of Revenue")
    plt.xlabel("Amount (millions)")
    plt.ylabel("Number of Players")

    plt.show()

def print_top_players(data: pd.DataFrame, count=10):
    ordered = data.sort_values("career_revenue")

    for i in range(1, count + 1):
        row = ordered.iloc[[-i]]
        name = row['name'].values[0]
        revenue = row['career_revenue'].values[0]
        print(f"{name}: ${revenue}")

if __name__ == "__main__":
    df_players = pd.read_csv("cleaned_players.csv")
    df_salaries = pd.read_csv("cleaned_salaries.csv")

    # show_salaries_histogram(df_salaries)
    # show_salaries_by_team(df_salaries)

    print_top_players(df_players)

    # show_career_revenue_distribution(df_players)


    pass

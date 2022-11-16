
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

def display_salary_over_time(salaries: pd.DataFrame):
    data = salaries.groupby("season_start").mean()

    years = data.index
    salaries = data["salary"] / 1e6
    adjusted_salaries = data["adjusted_salary"] / 1e6

    plt.plot(years.values, salaries.values, label="Salary")
    plt.plot(years.values, adjusted_salaries.values, label="Inflation adjusted")
    plt.title("Salaries over Time")
    plt.ylabel("Average Salary (millions)")
    # plt.xlabel("Year")
    plt.legend()
    plt.savefig("test")
    plt.show()

def plot_teams_box_plot(salaries: pd.DataFrame, col="salary"):
    plt.rc('figure', titlesize=20)
    plt.rc('axes', titlesize=16)
    plt.rc('xtick', labelsize=14)

    temp_data = salaries.copy()
    temp_data[col] /= 1e6

    salaries_by_team = temp_data.groupby("team")[col].apply(list).to_dict()

    teams_to_graph = {}

    # City of Houston
    teams_to_graph["Houston Rockets"] = salaries_by_team["Houston Rockets"]
    # Famous for having lots of super stars, but not super high
    teams_to_graph["Golden State Warriors"] = salaries_by_team["Golden State Warriors"]
    # Founded in 
    teams_to_graph["Brooklyn Nets"] = salaries_by_team["Brooklyn Nets"]
    # One of the newer teams, but not super high paid
    teams_to_graph["Memphis Grizzlies"] = salaries_by_team["Memphis Grizzlies"]
    # Bruh
    teams_to_graph["Kansas City Kings"] = salaries_by_team["Kansas City Kings"]

    fig = plt.figure()

    ax = fig.subplots()


    ax.boxplot(teams_to_graph.values(), showfliers=False)
    ax.set_xticklabels(map(lambda name: name.split()[-1], teams_to_graph.keys()), rotation='vertical')
    ax.set_ylabel("Salary (millions)")

    plt.title("Salaries by Team")

def plot_revenue_comparison(players: pd.DataFrame, col: str, title: str, xlabel: str, **kwargs):
    plt.scatter(players[col], players["adjusted_career_revenue"] / 1e6, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel("Career Revenue (millions)")
    plt.title(title)

if __name__ == "__main__":
    df_players = pd.read_csv("cleaned_players.csv")
    df_salaries = pd.read_csv("cleaned_salaries.csv")

    # show_salaries_histogram(df_salaries)
    # show_salaries_by_team(df_salaries)

    # print_top_players(df_players)

    # show_career_revenue_distribution(df_players)

    plt.rc('axes', titlesize=20)
    plt.rc('axes', labelsize=14)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    # plt.rc('ylabel', labelsize=14)
    plt.figure(dpi=200)

    # display_salary_over_time(df_salaries)
    # plot_teams_box_plot(df_salaries, "adjusted_salary")

    # df_main_players = df_players[df_players["career_g"] > 150]

    # plot_revenue_comparison(df_players, "career_pts", title="Importance of Scoring", xlabel="Points per Game", alpha=0.5, s=0.15)

    plot_revenue_comparison(df_players, "height", title="Importance of Height", xlabel="Height (in)", alpha=0.5, s=0.15)

    data = df_players.groupby(["height"]).mean()
    data["adjusted_career_revenue"] /= 1e6
    # plt.plot(data.index, data["adjusted_career_revenue"])
    print(df_players[df_players["height"] == 90])



    plt.savefig("test")
    plt.show()

    pass

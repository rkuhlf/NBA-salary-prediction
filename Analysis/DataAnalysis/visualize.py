
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from path_config import *
from Data.inputs import get_player_data, get_salary_data, get_inflation_data
from helpers import to_millions
from matplotlib_format import slides_format

def plot_salaries_histogram(data: pd.DataFrame):
    data = data.copy()

    data["salary"] = to_millions(data["salary"])

    plt.hist(data["salary"], bins=50)

    plt.title("Distribution of Salaries")
    plt.xlabel("Amount (millions)")
    plt.ylabel("Number of Salaries")

def plot_revenues_histogram(data: pd.DataFrame):
    data = data.copy()

    data["adjusted_career_revenue"] = to_millions(data["adjusted_career_revenue"])

    plt.hist(data["adjusted_career_revenue"], bins=50)

    plt.title("Distribution of Revenues")
    plt.xlabel("Amount (millions)")
    plt.ylabel("Number of Revenues")

def show_salaries_by_team(data: pd.DataFrame):
    # TODO
    data = data.copy()

    print(data.groupby("team").mean())

def plot_salary_over_time(salaries: pd.DataFrame):
    """
    Plots the average salary of each year compared to the inflation adjusted salary of each year across the league.
    """
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

def plot_player_salary(players: pd.DataFrame, salaries: pd.DataFrame, name: str):
    id = players[players["name"] == name]["id"].values[0]
    relevant_player = salaries[salaries["player_id"] == id]

    relevant_salaries = relevant_player["adjusted_salary"] / 1e6

    
    
    ax = plt.figure(dpi=200).gca()
    plt.plot(relevant_player["season_end"], relevant_salaries)
    
    plt.ylabel("Salary (millions)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title(f"{name} Salary")

def compare_team_salaries(salaries: pd.DataFrame, team1: str, team2: str, bins=20, relative=False):
    salaries = salaries.copy()
    salaries["adjusted_salary"] /= 1e6
    salaries_by_team = salaries.groupby("team")["adjusted_salary"].apply(list).to_dict()

    max_salary = max(np.max(salaries_by_team[team1]), np.max(salaries_by_team[team2]))

    bin_boundaries = range(0, int(max_salary) + 1)

    plt.ylabel("Frequency")
    plt.xlabel("Salary (millions)")

    team1_salaries = salaries_by_team[team1]
    team2_salaries = salaries_by_team[team2]
    weights1 = np.ones_like(team1_salaries)
    weights2 = np.ones_like(team2_salaries)

    if relative:
        weights1 = np.zeros_like(team1_salaries) + 1. / len(team1_salaries)
        weights2 = np.zeros_like(team2_salaries) + 1. / len(team2_salaries)

    plt.hist(team1_salaries, weights=weights1,
             label=team1, color='red', alpha=0.7, bins=bin_boundaries)
    plt.hist(team2_salaries, weights=weights2,
             label=team2, color='yellow', alpha=0.5, bins=bin_boundaries)
    
    plt.legend()





if __name__ == "__main__":
    players = get_player_data()
    salaries = get_salary_data()
    slides_format()

    # plot_salaries_histogram(salaries)
    
    # plot_revenues_histogram(players)
    # show_salaries_by_team(salaries)

    # print_top_players(players)

    # show_career_revenue_distribution(players)

    # display_salary_over_time(salaries)
    # plot_teams_box_plot(salaries, "adjusted_salary")

    # df_main_players = players[players["career_g"] > 150]

    # plot_revenue_comparison(players, "career_pts", title="Importance of Scoring", xlabel="Points per Game", alpha=0.5, s=0.15)

    # plot_revenue_comparison(players, "height", title="Importance of Height", xlabel="Height (in)", alpha=0.5, s=0.15)

    # data = players.groupby(["height"]).mean()
    # data["adjusted_career_revenue"] /= 1e6
    # plt.plot(data.index, data["adjusted_career_revenue"])
    # print(players[players["height"] == 90])

    # print((players["attended_college"] == True).sum())
    # print((players["attended_college"] == False).sum())

    # college_average = players[players["attended_college"] == True]["adjusted_career_revenue"].mean() / 1e6
    # no_college_average = players[players["attended_college"] == False]["adjusted_career_revenue"].mean() / 1e6

    # plt.bar(["College", "No College"], [college_average, no_college_average])
    # plt.title("Should I Go to College?")
    # plt.ylabel("Average Earnings (millions)")

    # plot_player_salary(players, salaries, "Michael Jordan")
    # plot_player_salary(players, salaries, "Kevin Garnett")
    # compare_team_salaries(salaries, "Brooklyn Nets", "Houston Rockets")
    # plt.title("The Nets Never Paid Low")

    # compare_team_salaries(salaries, "Brooklyn Nets", "Houston Rockets")
    # plt.title("The Grizzlies Never Paid Low")

    plt.savefig("test")
    plt.show()

    pass

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def slides_format():
    plt.rc('axes', titlesize=20)
    plt.rc('axes', labelsize=14)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    # plt.rc('ylabel', labelsize=14)
    # plt.figure(dpi=200)

def integer_axis(ax):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
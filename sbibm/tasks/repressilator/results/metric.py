import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('c2st_stats.csv')
plt.rcParams['font.size'] = 20

# Get the unique algorithms and the number of unique algorithms
algorithms = df['Algorithm'].unique()
num_algorithms = len(algorithms)

# Set up the subplots
fig, axs = plt.subplots(1, num_algorithms, figsize=(6*num_algorithms, 6), sharey=True)

# Define a custom palette with repeated colors for the first and third, and second and fourth algorithms.
cubehelix_palette = sns.cubehelix_palette(10, start=2, rot=1, dark=0.3, light=0.8) 
#palette = sns.color_palette(["#1f77b4", "#ff7f0e", "#1f77b4", "#ff7f0e"])
 # generate cubehelix palette
palette = [cubehelix_palette[-1],cubehelix_palette[4],cubehelix_palette[-1],cubehelix_palette[4]]
# Get the global minimum and maximum 'Simulation Budget' for consistent x-axis
global_xmin = df['Simulation Budget'].min()
global_xmax = df['Simulation Budget'].max()

# Extract the required columns from the DataFrame
for i, algorithm in enumerate(algorithms):
    df_algo = df[df['Algorithm'] == algorithm]
    x = df_algo['Simulation Budget']
    y = df_algo['Mean C2ST']
    std_dev = df_algo['Standard Deviation C2ST']

    # Calculate the upper and lower bounds for the error bars (95% confidence intervals)
    yerr = 1.96 * std_dev / np.sqrt(y.size)

    # log scale for x-axis, linear scale for y-axis
    axs[i].set_xscale('log')

    # Plot the data with Seaborn
    sns.lineplot(x=x, y=y, ax=axs[i], color=palette[i % len(palette)], marker="o", ci=95)
    # Plot errorbars
    axs[i].errorbar(x, y, yerr=yerr, fmt='o', color=palette[i % len(palette)])

    # Set labels with bigger font
    axs[i].set_xlabel('Simulation Budget', fontsize=30)
    axs[i].set_title(algorithm, fontsize=30)
    axs[i].tick_params(axis='both', which='major', labelsize=25)

    # Set x and y axis limits
    axs[i].set_xlim([global_xmin / 1.1, global_xmax * 1.1])  # Add some padding to the x-axis
    axs[i].set_ylim([0.5 - 0.05, 1.0 + 0.1])  # Set y-axis limits to slightly beyond 0.5 to 1.1 to avoid markers on x-axis

    if i == 0:  # only set the y label for the first subplot
        axs[i].set_ylabel('C2ST', fontsize=30)

    # Remove spines and ticks
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].tick_params(which='both', bottom=False, left=False)

    # Ensure y-ticks are not in scientific notation
    fmt = FuncFormatter(lambda x, _: '{:.1f}'.format(x))
    axs[i].yaxis.set_major_formatter(fmt)

# Set darkness for all gridlines
for ax in axs:
    ax.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.75)

plt.tight_layout()
plt.savefig("c2st_all_algo.pdf")
plt.show()
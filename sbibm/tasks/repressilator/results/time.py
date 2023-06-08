import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('c2st_stats.csv')
# Set larger font size
plt.rcParams['font.size'] = 20

# Get the unique algorithms and the number of unique algorithms
algorithms = df['Algorithm'].unique()
num_algorithms = len(algorithms)

# Set up the subplots
fig, axs = plt.subplots(1, num_algorithms, figsize=(6*num_algorithms, 6), sharey=True)

# Define Seaborn style
sns.set_style("whitegrid")

# Define a custom palette with repeated colors for the first and third, and second and fourth algorithms.
cubehelix_palette = sns.cubehelix_palette(10, start=2, rot=1, dark=0.3, light=0.8) 
#palette = sns.color_palette(["#1f77b4", "#ff7f0e", "#1f77b4", "#ff7f0e"])
 # generate cubehelix palette
palette = [cubehelix_palette[-1],cubehelix_palette[4],cubehelix_palette[-1],cubehelix_palette[4]]
# Get the global minimum and maximum 'Simulation Budget' for consistent x-axis
global_xmin = df['Simulation Budget'].min()
global_xmax = df['Simulation Budget'].max()

# Iterate over each unique algorithm
for i, algorithm in enumerate(algorithms):
    df_algo = df[df['Algorithm'] == algorithm]
    x = df_algo['Simulation Budget']
    y = df_algo['Mean Execution Time'] / 60  # Here is the change
    std_dev = df_algo['Standard Deviation Time'] / 60  # Assuming this is the name of the column

    # Calculate the upper and lower bounds for the error bars (95% confidence intervals)
    yerr = 1.96 * std_dev / np.sqrt(y.size)

    # Log scale
    axs[i].set_yscale('log')
    axs[i].set_xscale('log')

    # Plot the data with Seaborn
    sns.lineplot(x=x, y=y, ax=axs[i], color=palette[i % len(palette)], marker="o", ci=95)

    # Plot error bars
    axs[i].errorbar(x, y, yerr=yerr, fmt='o', color=palette[i % len(palette)])

    # Set labels
    axs[i].set_xlabel('Simulation Budget', fontsize=30)
    if i == 0:  # Only set the y label for the first subplot
        axs[i].set_ylabel('Execution Time', fontsize=30)
    axs[i].set_title(algorithm, fontsize=30)
    axs[i].tick_params(axis='both', which='major', labelsize=25)

    # Set x and y axis limits
    axs[i].set_xlim([global_xmin / 1.1, global_xmax * 1.1])  # Add some padding to the x-axis
    axs[i].set_ylim([0.1, 10000 + 0.1])  # Set y-axis limits to slightly beyond 0.5 to 1.1 to avoid markers on x-axis

    # Remove spines and ticks
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].tick_params(which='both', bottom=False, left=False)

    axs[i].yaxis.set_major_formatter(ScalarFormatter(useMathText=False))

    # Set y-axis ticks
    axs[i].set_yticks([1, 10, 100, 1000])

    # Remove vertical grid lines
    axs[i].xaxis.grid(False)

    # Set darkness for all horizontal gridlines
    axs[i].yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.75)

plt.tight_layout()
plt.savefig("time_noisy.pdf")
plt.show()

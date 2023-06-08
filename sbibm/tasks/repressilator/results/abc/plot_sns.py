import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

def posterior_plot(data:str):
    df = pd.read_csv(data)
    df=df[df['Group']=='p'] 
    plt.figure(figsize=(8, 8)) 
    # Create PairGrid 
    g = sns.PairGrid(df, vars=df.columns[:12], corner=True)
    g = g.map_diag(sns.kdeplot)
    g = g.map_offdiag(sns.kdeplot)
    # Setting x and y axis limits
    for ax in g.axes.flatten():
        if ax is not None:
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    plt.suptitle('Pairplot of Posterior Samples', fontsize=50)
    # # Create new legend
    # legend_elements = [Patch(facecolor='blue', edgecolor='blue', label='p'),
    #                    Patch(facecolor='orange', edgecolor='orange', label='r')]
    # plt.legend(handles=legend_elements, title='Group', loc='upper right')
    plt.show()

posterior_plot('df_posterior_2_1000_smc_abc.csv')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import MaxNLocator

def posterior_plot(data_list,namefig):
    color_dict = {'r': '#f5b59a'}  # set color for 'r'
    cubehelix_palette = sns.cubehelix_palette(10, start=2, rot=1, dark=0.3, light=0.8)  # generate cubehelix palette
    color_map = {0: -1, 1: 4, 2: -1, 3: 4}  # map of column index to palette color

    num_vars = None
    fig, ax = plt.subplots(3, len(data_list) // 3, figsize=(8 * len(data_list) // 3, 24))  # 3 rows, each algorithm has a column

    for idx in range(len(data_list) // 3):  # Loop over each algorithm
        for budget_idx in range(3):  # Loop over each budget
            data = data_list[idx * 3 + budget_idx]
            df = pd.read_csv(data)
            df = df[df['Group'].isin(['p', 'r'])]  # Include both 'p' and 'r' groups

            if num_vars is None:
                num_vars = df.columns[:1]  # selecting the first column

            color_dict['p'] = cubehelix_palette[color_map[idx]]  # set color for 'p' group from the cubehelix palette

            for j, var in enumerate(num_vars):
                for group in ['p', 'r']:
                    sns.kdeplot(df[df['Group']==group][var], ax=ax[budget_idx, idx], label=group, color=color_dict[group], fill=True, alpha=0.5)
                ax[budget_idx, idx].set_xlim(-3, 3)
                ax[budget_idx, idx].xaxis.set_major_locator(MaxNLocator(nbins=3))
                ax[budget_idx, idx].yaxis.set_major_locator(MaxNLocator(nbins=3))
                ax[budget_idx, idx].tick_params(axis='both', which='major', labelsize=30)
                ax[budget_idx, idx].xaxis.label.set_size(40)
                ax[budget_idx, idx].yaxis.label.set_size(40)
                if idx != 0:
                    ax[budget_idx, idx].set_ylabel('')  # Set empty string as the y-axis title

    plt.tight_layout()
    plt.savefig("notnoise.png") #was output

    plt.show()

data_files = ['df_posterior_3_1000_rej_abc.csv', 'df_posterior_3_10000_rej_abc.csv', 'df_posterior_3_100000_rej_abc.csv',
              'df_posterior_3_1000_npe.csv', 'df_posterior_3_10000_npe.csv', 'df_posterior_3_100000_npe.csv',
              'df_posterior_3_1000_smc_abc.csv', 'df_posterior_3_10000_smc_abc.csv', 'df_posterior_3_100000_smc_abc.csv',
              'df_posterior_3_1000_snpe.csv', 'df_posterior_1_10000.snpe.csv','df_posterior_1_1000.snpe.csv']
posterior_plot(data_files, 'notnoise')
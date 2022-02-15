import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import math 

import src.visualizations.plot_utils as plt_utils

df = pd.read_csv("data/processed/sf_data_processed.csv")

def time_plot(date_tuple = None, label = None, ylim = (0,140)):

    # number of crimes on each date 
    if label is not None: 
        n_crimes = (df
            .query(f"label == '{label}'")
            .groupby(["district", "date"])
            .agg(N = ("id", "count"))
            .unstack(fill_value=0)
            .stack()
            .reset_index()
        )
    else: 
        n_crimes = (df
            .groupby(["district", "date"])
            .agg(N = ("id", "count"))
            .unstack(fill_value=0)
            .stack()
            .reset_index()
        )
        label = ''

    if date_tuple is not None: 
        date_start = date_tuple[0]
        date_end = date_tuple[1]
        n_crimes = n_crimes.query(f"date >= '{date_start}' & date <= '{date_end}'")

    fig, ax = plt.subplots(1,1, figsize = (15,5))
    cp = plt_utils.custom_palette(n_colors=3)
    for i, district in enumerate(df.district.unique()):
        plot_df = n_crimes.query(f"district == '{district}'")
        plt.plot(plot_df.date, plot_df.N, color = cp[i], label = f"{district}")

        if i == 2: 
            n_ticks = 60
            step = math.floor((len(plot_df)-1)/n_ticks)
            idx_int = np.arange(0, len(plot_df), step = step)
            ax.set_xticks(plot_df.date.iloc[idx_int])
            ax.set_xticklabels(plot_df.date.iloc[idx_int], rotation = 90)
    plt.legend()
    plt.title(f"Number of {label} Crimes per Day")
    plt.ylabel("Number of Crimes")
    plt.xlabel("Date")
    plt.ylim(ylim)

# Crime vs. time all types of crime 
time_plot(None)
time_plot(("2013-01-01", df.date.max()))
time_plot((df.date.min(), "2012-12-31"))

# Crime vs. time violent crime 
time_plot(None, label = 'violent')
time_plot(("2013-01-01", df.date.max()), label = 'violent')
time_plot((df.date.min(), "2012-12-31"), label = 'violent')


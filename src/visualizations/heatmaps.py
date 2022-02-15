import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import src.visualizations.plot_utils as plt_utils
cp = plt_utils.custom_palette(n_colors=6)

init_df = pd.read_csv("data/processed/sf_data_processed.csv")

df = init_df.query("label == 'violent'")

# At what time of day does the crimes occur?
district = 'sunnydale' 
df_district = (df.query(f'district == "{district}"')
    .groupby(['label', 'hour'])
    .agg(N = ('id', 'count')) 
    .reset_index())

plot_df = df_district.pivot(index = "label", columns = "hour", values = "N")
sns.heatmap(plot_df)
plt.title(f'{district}')

# Heatmap of crime type for hours in a day across districts 
def heatmap_label_across_districts(label, fraction = True):
    # Heatmap of crime label across districts  
    n_crimes = (df.query(f'label == "{label}"')
        .groupby(['district'])
        .agg(N = ('id', 'count'))).to_dict()['N']

    df_label = (df.query(f'label == "{label}"')
        .groupby(['district', 'hour'])
        .agg(N = ('id', 'count')) 
        .reset_index()
        .assign(
            N_perc = lambda x: [x.N[i]/n_crimes[x.district[i]] for i in range(len(x.N))]))

    if fraction: 
        plot_df = df_label.pivot(index = "district", columns = "hour", values = "N_perc")
    else: 
        plot_df = df_label.pivot(index = "district", columns = "hour", values = "N") 
    sns.heatmap(plot_df)
    plt.title(f'Label: {label}')

heatmap_label_across_districts('violent', False)

# At what time of day and weekday does the crimes occur in the three districts?
# Only violent crime
def heatmap_by_weekdays(district, vmax = 500):
    df_district = (df.query(f'district == "{district}"')
            .groupby(['label', 'hour', 'weekday'])
            .agg(N = ('id', 'count')) 
            .reset_index())
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

    fig, axn = plt.subplots(7, 1, sharex=True, sharey=True, figsize = (10, 8))
    cbar_ax = fig.add_axes([.91, .3, .03, .4])

    for i, ax in enumerate(axn.flat):
        plot_df = df_district.query(f"weekday == '{days[i]}'") 
        d = plot_df.pivot(index='label', columns="hour", values="N")
        s = sns.heatmap(d, ax=ax,
                    cbar=i == 0,  yticklabels=False,
                    vmin=0, vmax = vmax, cmap = 'rocket_r', 
                    cbar_ax=None if i else cbar_ax)
        s.set_ylabel(days[i], rotation = 90)
        if i < 6: 
            s.set_xlabel('')
    plt.suptitle(f'Violent crime in {district}')

heatmap_by_weekdays('tenderloin', vmax = 500)
heatmap_by_weekdays('sunnydale', vmax = 30)
heatmap_by_weekdays('mission', vmax = 400)

#%%
# At what time of day and weekday does the crimes occur in the three districts?
# all crime
def heatmap_by_weekdays_all_crime(district, vmax = 500):
    df_district = (init_df.query(f'district == "{district}"')
            .groupby(['hour', 'weekday'])
            .agg(N = ('id', 'count')) 
            .reset_index())
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

    fig, axn = plt.subplots(7, 1, sharex=True, sharey=True, figsize = (10, 8))
    cbar_ax = fig.add_axes([.91, .3, .03, .4])

    for i, ax in enumerate(axn.flat):
        plot_df = df_district.query(f"weekday == '{days[i]}'") 
        d = plot_df.pivot(index='weekday', columns="hour", values="N")
        s = sns.heatmap(d, ax=ax,
                    cbar=i == 0,  yticklabels=False,
                    vmin=0, vmax = vmax, cmap = 'rocket_r', 
                    cbar_ax=None if i else cbar_ax)
        s.set_ylabel(days[i], rotation = 90)
        if i < 6: 
            s.set_xlabel('')
    plt.suptitle(f'Crime in {district}')

heatmap_by_weekdays_all_crime('tenderloin', vmax = 3500)
heatmap_by_weekdays_all_crime('sunnydale', vmax = 150)
heatmap_by_weekdays_all_crime('mission', vmax = 2500)


# %%

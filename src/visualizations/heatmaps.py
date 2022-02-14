import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import src.visualizations.plot_utils as plt_utils
cp = plt_utils.custom_palette(n_colors=6)

df = pd.read_csv("data/processed/sf_data_processed.csv").query("label == 'violent'")

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

# At what time of day does the crimes occur? 
# plus also considering weekdays vs. weekends
def heatmap_by_district_and_weekday(district, vmax):
    df_district = (df.query(f'district == "{district}"')
        .groupby(['label', 'hour', 'weekday'])
        .agg(N = ('id', 'count')) 
        .reset_index())

    def draw_heatmap(*args, **kwargs):
        data = kwargs.pop('data')
        d = data.pivot(index=args[1], columns=args[0], values=args[2])
        sns.heatmap(d, **kwargs, cmap = 'rocket_r')

    cbar_ax = g.fig.add_axes([100, 10, 4, 7])
    fg = sns.FacetGrid(df_district, 
                        row='weekday',
                        sharey=True,
                        aspect = 6,
                        height = 1, 
                        row_order=['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'])
    fg.map_dataframe(draw_heatmap, 'hour', 'label', 'N', cbar_ax = cbar_ax, square = False, cbar = True, vmin = 0, vmax = vmax)
    
    fg.fig.suptitle(f"District: {district}")


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


heatmap_by_district_and_weekday('tenderloin',vmax = 500)
heatmap_by_district_and_weekday('sunnydale', vmax = 30)
heatmap_by_district_and_weekday('mission', vmax = 400)


# Heatmap by label and weekday
# TODO use the same color bar 
def heatmap_by_label_and_weekday(label, vmin=0, vmax=0):
    n_crimes = (df.query(f'label == "{label}"')
        .groupby(['district'])
        .agg(N = ('id', 'count'))).to_dict()['N']

    df_label = (df.query(f'label == "{label}"')
        .groupby(['district', 'hour', 'weekday'])
        .agg(N = ('id', 'count')) 
        .reset_index()
        .assign(
            N_perc = lambda x: [x.N[i]/n_crimes[x.district[i]] for i in range(len(x.N))]))

    def draw_heatmap(*args, **kwargs):
        data = kwargs.pop('data')
        d = data.pivot(index=args[1], columns=args[0], values=args[2])
        sns.heatmap(d, **kwargs)

    fg = sns.FacetGrid(df_label, row='weekday', sharey=True, aspect = 4, height = 2.5)
    fg.map_dataframe(draw_heatmap, 'hour', 'district', 'N', cbar=True, square = False)
    fg.fig.subplots_adjust(top=0.9)
    fg.fig.suptitle(f"Label: {label}")

heatmap_by_label_and_weekday('violent')
heatmap_by_label_and_weekday('other')
heatmap_by_label_and_weekday('prostitution')
heatmap_by_label_and_weekday('intoxication')


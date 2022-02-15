import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import src.visualizations.plot_utils as plt_utils
cp = plt_utils.custom_palette(n_colors=4, specific_col_idx=[0, 9, 6, 4])

df = pd.read_csv("data/processed/sf_data_processed.csv")
#%%
# amount of recorded crimes in the three disctricts 
df.groupby(['district']).agg(N = ('id', 'count'))

# How is the type of crime in the three districts 
df_dict = dict()
for i, dist in enumerate(df.district.unique()):
    tmp = df.query(f"district == '{dist}'")
    N_df = (tmp.groupby(["label"])
        .agg(N = ('id', "count"))
        .reset_index()
        .assign(N_perc = lambda x: x.N/x.N.sum(),
                N_people = lambda x: x.N.sum()))
    
    df_dict[dist] = N_df

dist_crime = pd.concat(df_dict, names = ['district']).reset_index().drop(columns='level_1')

g = sns.catplot(
    data = dist_crime, 
    kind = 'bar',
    x = "district", 
    y = 'N',
    hue = 'label', palette = cp)
g.set(title = 'Number of Recorded Crimes within District',
    xlabel = 'District')

g = sns.catplot(
    data = dist_crime, 
    kind = 'bar',
    x = "district", 
    y = 'N_perc',
    hue = 'label', palette = cp)
g.set(ylim=(0,1), 
      title = 'Fraction of Recorded Crimes within District',
      xlabel='District',
      ylabel='')

#%%
# How is crime category in each district? 
def plot_N_crime_category(df, fraction = False):
    df_dict_cat = dict()
    for i, dist in enumerate(df.district.unique()):
        tmp = df.query(f"district == '{dist}'")
        N_df = (tmp.groupby(["category", "label"])
            .agg(N = ('id', "count"))
            .reset_index()
            .assign(N_perc = lambda x: x.N/x.N.sum(),
                    N_people = lambda x: x.N.sum()))
        
        df_dict_cat[dist] = N_df

    dist_crime_cat = pd.concat(df_dict_cat, names = ['district']).reset_index().drop(columns='level_1')

    if fraction: 
        g = sns.catplot(
            data = dist_crime_cat, 
            kind = 'bar',
            x = "N_perc", 
            y = 'category',
            col = 'district', 
            hue = 'label', legend=False, dodge = False, 
            palette=plt_utils.custom_palette(n_colors=4, specific_col_idx=[0, 9, 6, 4]))
    else:
        g = sns.catplot(
            data = dist_crime_cat, 
            kind = 'bar',
            x = "N", 
            y = 'category',
            col = 'district', 
            hue = 'label',
            legend=False,
            dodge = False, 
            palette=plt_utils.custom_palette(n_colors=4, specific_col_idx=[0, 9, 6, 4]))

plot_N_crime_category(df, fraction = True)
plot_N_crime_category(df, fraction = False)

# across 5 years sunnydale sees no remarkable crime rate compared to the others
# but they have similar patterns of violent crime (percentage wise)
df_violent = df.query("label == 'violent'")
plot_N_crime_category(df_violent, fraction = True)
plot_N_crime_category(df_violent, fraction = False)






# %%

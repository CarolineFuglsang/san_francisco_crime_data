import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

# loading raw data 
raw = pd.read_csv("./data/raw/sf_data.csv", sep=';')
raw_districts = pd.read_csv("./data/raw/sf_districts.csv", sep=';')

raw.head(10)

# ---------------------------------------------------------------
# Category vs. Description 
raw.category.nunique()
(raw.groupby(['category'], as_index = False)
    .agg(N = ('id', 'count'))
)

raw.description.nunique()
gb_desc = (raw.groupby(['description'], as_index = False)
    .agg(N = ('id', 'count'))
    .sort_values(by = 'N')
)

# 46% of descriptions sees less than 100 observations 
sns.histplot(data = gb_desc.query('N<=5000'), x = 'N')
gb_desc.query('N <= 100').shape[0]

# are descriptions uniquely contained in a category? 
desc_dict = dict()
for i, cat in enumerate(raw.category.unique()):
    desc_dict[cat] = raw.query(f"category == '{cat}'").description.unique()

intersect_df = pd.DataFrame(columns = ["cat_1", "cat_2", "intersection"])
for i, cat_1 in enumerate(raw.category.unique()):
    for j, cat_2 in enumerate(raw.category.unique()):
        if cat_1 == cat_2: 
            continue
        cat_1_descs = set(desc_dict[cat_1])
        cat_2_descs = set(desc_dict[cat_2])
        intersect = cat_1_descs.intersection(cat_2_descs)

        if len(intersect) != 0: 
            tmp = pd.DataFrame({'cat_1': cat_1, 
                                'cat_2': cat_2,
                                'intersection': list(intersect)})
            intersect_df = pd.concat([intersect_df, tmp])

# 9 incidences where a description is within more than one category the incidences have been found twice but in the interest of time I choose not to correct this for the time being.

# Idea for coding it better: Group by followed by comparison to descriptions in all categories with a higher index than the current category: 
(raw.groupby(["category"])
    .agg(desc_unique = ("description", lambda x: x.unique()))
    .reset_index()
)

#--------------------------------------------------------------------
# label vs. category 
# the label is unique within a category
(raw.groupby(["category"])
    .agg(label_unique = ("label", lambda x: x.unique()))
    .reset_index()
)

# -----------------------------------------------------------------
# The amount of districts 
raw_districts.district.nunique()



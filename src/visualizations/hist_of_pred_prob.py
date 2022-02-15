import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
nndf = pd.read_csv("data/predictions/NN_pred.csv")
logregdf = pd.read_csv("data/predictions/log_reg.csv")



v_df = logregdf.query("is_violent == True")
df = logregdf.query("is_violent == False")


fig, ax = plt.subplots(1, 1, figsize = (6,4))
sns.histplot(data = logregdf, x = 'log_reg_prob', hue = "is_violent", ax = ax)
ax.set_xlim((0.1, 0.3))

fig, ax = plt.subplots(1, 1, figsize = (6,4))
sns.histplot(data = nndf, x = 'nn_prob', hue = "is_violent")
ax.set_xlim((0.1, 0.3))

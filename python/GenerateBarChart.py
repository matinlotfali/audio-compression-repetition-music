import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('tmp/output.csv')
df = df.drop(columns=['name'])
yvals = df.mean(axis=0).values
y_std = df.std() / np.sqrt(len(df)) * 1.96
plt.bar(df.columns, yvals, yerr=y_std, width=0.5, capsize=10)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('tmp/bar.png')
plt.close()

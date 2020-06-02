import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import scipy.stats as st


df = pd.read_csv("Admission_Predict_LA.csv")
df.drop(columns=['Serial No.'], inplace=True)
headers = list(df.columns)
target = headers.pop()
X = list(df[headers].values)
y = list(df['ChanceofAdmit'].values)

r, p = st.pearsonr(df['SOP'], df['ChanceofAdmit'])

bins = [1.0, 2.5, 3.5, 5.0]
labels = ["poor(1.0~2.5)", 'avg(2.5~3.5)', 'good(4.0~5.0)']
df['SOP_bins'] = pd.cut(df['SOP'], bins, labels=labels, include_lowest=True)

sns.boxplot(x="SOP_bins", y="ChanceofAdmit", data=df)
plt.show()


'''
print(f"Pearson coefficient: {r}")
print(f"p-value: {p}")


sns.regplot(x="SOP", y="ChanceofAdmit", data=df)
plt.show()'''

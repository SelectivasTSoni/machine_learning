#!/usr/bin/env python3

# diabetes_analysis.py

print("good morning")

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# read the given dataset
d = pd.read_csv('diabetes_data.csv')


print(d.head(10), '\n')
print(d.tail(10), '\n')
print('The shape of the dataset is: ', d.shape, '\n')
print('Describe: \n', d.describe())

# I am not sure why this returns None.
print('Info: \n', d.info(verbose=True), '\n')

# outcome indicates diabetic or non-diabetic
# get series
dia = d.loc[d['Outcome']]

# get counts
dia.value_counts()

# any missing values? No.
d.isna()

# get median
median = d['SkinThickness'].median()

# assign
d['SkinThickness'] = median
print(d['SkinThickness'])

# Get mean age
print(d['Age'].mean())

# Highest glucose
d.nlargest(5, 'Glucose', keep='first')

d['LogPedigreeFunction'] = np.log2(d['DiabetesPedigreeFunction'])
d.head(5)

# multiply the columns
d['Age_Insulin'] = d.Age * d.Insulin

# show result
d.head(5)

# make bar chart to show number of diabetic and non-diabetic patients
plt.figure(figsize=(8, 5))
sns.countplot(x='Outcome', data=d)
plt.title('Distribution of Outcome')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['No Diabetes (0)', 'Diabetes (1)'])
plt.show()

sns.scatterplot(data=d, x="Glucose", y="Insulin")




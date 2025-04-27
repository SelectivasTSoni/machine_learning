#!/usr/bin/env python3

# pd_basics.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

s = pd.Series([1, 2, 5, np.nan, 6, 8])
print(s)

dates = pd.date_range('20231009', periods=10)
print(dates)

# make pandas dataframe, df
# the np.random.randn(6,4) the 6,4 is the shape. 
# - We used a range of 10 dates, so the 6 needs be a 10
# - note the 4 must match the length of the number of 
# columns=list("....") entries
# error: df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('abcd'))
# corrected:
df = pd.DataFrame(np.random.randn(10, 4), index=dates, columns=list('abcd'))
print(df)

# an alternative way to create a dataframe is to pass a dict of objects
# that can be converted to series-like data-structure
df2 = pd.DataFrame({'a': 1.,
                    'b': pd.Timestamp('20231009'),
                    'c': pd.Series(1, index=list(range(4)),dtype='float'),
                    'd': np.array([3]*4,dtype='int32'),
                    'e': pd.Categorical(["test", "train", "test", "train"]),
                    'f': 'foo',
                   })

print(df2.dtypes)

# #tab completion, df2.<TAB> will show a list of df attributes that can
# # be accessed with the dot operator
# print(df2.abs())
# this shows the column 'a' we created
print(df2.a)

print(df.head())

print(df.tail(3))

print(df.index)

print(df.columns)
print(df.values)

# generate summary statistics for each column
print(df.describe())

# transpose data (switch rows and columns)
print(df.T)

# some sorting
print(df.sort_index(axis=1, ascending=False))
print(df.sort_index(axis=1, ascending=True))
print(df.sort_values(by='c'))

print(df.loc[dates[0]])
print(df.sort_index(axis=1, ascending=True))

print(df.loc[dates[0]])
print(df.loc[dates[3]])

# this is ``get data from all of one axis (dates), with two features from the other axis''
print(df.loc[:,['a','b']])

# we can slice axes to get whatever data locations we want
print(df.loc['20231012':'20231015', ['a','b']])

# dimension reduction
print(df.loc['20231014',['a','c']])
# we had two dimensions (date and attributes), now we have one (attribute).

# for getting scalar values
print(df.loc[dates[0],'a'])
print(df.loc[dates[0],'d'])

# select by position
print(df)
# this is row 3
print(df.iloc[3])
# this is row 1, 2, 4 and column 0, 2, We are on zero-based indexes
print(df.iloc[[1,2,4], [0,2]])

# values less than zero come back NaN
print(df[df>0])

# make a copy of the df
df2 = df.copy()

# append a column
df2['e'] = ['one', 'two', 'two', 'three', 'four', 'four', 'five', 'six', 'seven', 'eight']
print(df2)

# filter rows where 'e' column is either 'one' or 'three'
print(df2[df2['e'].isin(['one', 'three'])])

# setting a new column using the same date range 
# We stored this range as a variable up at input[5] called dates
sl = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=dates)
print(sl)

# we add a column for sl by assignment
df2['f'] = sl
print(df2)




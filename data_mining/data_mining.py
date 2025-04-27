#!/usr/bin/env python3

import pandas as pd
import missingno as msno

ds = pd.read_csv("data_mining_data.csv")

print(ds.head(5))
print(ds.info())
print(ds.shape)

sender_nulls = ds["Sender domain"].isnull().sum()
print("\n")
print("null values in 'Sender domain': {}".format(sender_nulls))

# Identify unique values within the features
da_values = ds['Delivery action'].unique().tolist()
threats_values = ds['Threats'].unique().tolist()
phish_con_lvl_values = ds['Phish confidence level'].unique().tolist()
sender_dom_values = ds['Sender domain'].unique().tolist()
recip_values = ds['Recipient tags'].unique().tolist()

# Display the number of domains (unique values) in the Sender domain column
print("Unique values in 'Sender domain' column: {}".format(len(sender_dom_values)))


# Do we have any missing values?
missing = ds.isnull().sum().sum()
print("Number of missing values: {}".format(missing))


## Plot the no. of missing values (i.e., count) across all the features
#msno.matrix(ds)
msno.bar(ds)


# Drop the rows where at least one element is missing.
ds_dropped_na = ds.dropna()

## Display your dataset after dropping missing values
ds_dropped_na.head(5)
ds_shape = ds.shape
ds_dropped_na_shape = ds_dropped_na.shape

print("shape of ds: {}".format(ds_shape))
print("shape of ds_dropped_na: {}".format(ds_dropped_na_shape))


# Modify "Recipient tags" feature so values are consistent
s = ds['Recipient tags']
s_2 = s.str.split(pat="|", expand=True)
u_list = s_2[0].unique().tolist()
print(u_list)
print("Unique values in 'u_list' column: {}".format(len(u_list)))

df_recip_tags_copy = s_2.copy()
cleaned = s_2.replace('\|\d*', '', regex=True)
print(cleaned)

cleaned_again = cleaned.replace('\-\d*', '', regex=True)
print(cleaned_again)

# One-hot encoding the "Threats" and "Recipient tags"
ds_encoded = pd.get_dummies(ds, columns=['Threats', 'Recipient tags'])

#data1 = pd.get_dummies(data=ds, columns = ['Threats'])
#data2 = pd.get_dummies(data=ds, columns = ['Recipient tags'])


## Calculate the top 5 domains that are generating malicious emails
print(ds['Sender domain'].value_counts().head(5))

# Filtering the dataset only for 'high' phish confidence levels
# get high confidence level count
ds_high_confidence = ds[ds['Phish confidence level'] == 'High']

## Calculate the no. of unique values for the "Sender domain" feature with 'high' phish confidence levels

print('Number of unique senders:'.format(ds['Sender domain'].unique()))

print('Number of Phish_confidence High values: {}'.format(len(ds_high_confidence)))
print('Number of Unique recipient values: {}'.format(len(recip_values)))

# get ratio of high fishing confidence level and total domains

# total number of domains (repeated from way up top)
sender_dom_values = ds['Sender domain'].unique().tolist()

ratio_1 = len(ds_high_confidence) / len(sender_dom_values)
ratio_2 = len(sender_dom_values) / len(ds_high_confidence)

print("High confidence count: {}".format(len(ds_high_confidence)))

print("total domains count: {}".format(len(sender_dom_values)))

print("Ratio of high phishing confidence level to total domains: {}".format(ratio_2))

# need unique recipient tag values
ds_unique_recipient_tags = ds['Recipient tags'].unique()

print(ds_unique_recipient_tags)

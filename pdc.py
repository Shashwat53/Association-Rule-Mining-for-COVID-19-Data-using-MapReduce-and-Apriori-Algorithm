# -*- coding: utf-8 -*-
"""PDC.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zRr_VGG5FW98bHjDIa-SKFJnxvsSB03x
"""

# Step 1: Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Loading the dataset
df = pd.read_csv('COVID_Optimisation.csv', header=None)

# Step 3: Exploring the dataset
print(df.head())

# Step 4: Data cleaning and transformation
df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x) # Converting all values to lowercase
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x) # Removing leading and trailing whitespaces
df = df.drop_duplicates() # Removing duplicates

# Step 5: Formatting the data
transaction_list = []
for i in range(0, len(df)):
    transaction = []
    for j in range(0, len(df.columns)):
        if str(df.values[i, j]) != 'nan':
            transaction.append(str(df.values[i, j]))
    transaction_list.append(transaction)

# Step 6: Saving the cleaned and formatted dataset
cleaned_df = pd.DataFrame(transaction_list)
cleaned_df.to_csv('COVID_Optimisation.csv', index=False, header=False)

import csv
from collections import defaultdict

# Mapper function
def mapper(row):
    items = ",".join(row).strip().split(",")
    return [(item, 1) for item in items]

# Reducer function
def reducer(item, values):
    count = sum(values)
    return (item, count)

# Read the CSV file and process the data
with open('COVID_Optimisation.csv', 'r') as file:
    reader = csv.reader(file)
    data = []
    for row in reader:
        data.extend(mapper(row))
    groups = defaultdict(list)
    for key, value in data:
        groups[key].append(value)
    results = map(lambda x: reducer(x[0], x[1]), groups.items())

# Print the count of each item in the dataset
for item, count in results:
    print(item, count)

# DEPENDENCIES
import pandas as pd 
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

# df.read_csv('filepath',headers=None)
df = pd.read_csv("COVID_Optimisation.csv")

# Let's have a look at the first few rows in our dataframe.
df.head()

# replace all the NaN values with ‘’ and use inplace=True to commit the changes permanent into the dataframe
df.fillna('',axis=1,inplace=True)
df.head()

# importing module
import numpy as np
# Gather All Items of Each Transactions into Numpy Array
transaction = []
for i in range(0, df.shape[0]):
    for j in range(0, df.shape[1]):
        transaction.append(df.values[i,j])
# converting to numpy array
transaction = np.array(transaction)
#  Transform Them a Pandas DataFrame
dF = pd.DataFrame(transaction, columns=["items"]) 
# Put 1 to Each Item For Making Countable Table, to be able to perform Group By
dF["incident_count"] = 1 
#  Delete NaN Items from Dataset
indexNames = dF[dF['items'] == "nan" ].index
dF.drop(indexNames , inplace=True)
# Making a New Appropriate Pandas DataFrame for Visualizations  
dF_table = dF.groupby("items").sum().sort_values("incident_count", ascending=False).reset_index()
#  Initial Visualizations

dF_table.iloc[1:10].style.background_gradient(cmap='Blues')

import matplotlib.pyplot as plt

# sort the DataFrame by incident_count in descending order and select the top 10 items
top_20_items = dF_table.sort_values("incident_count", ascending=False).head(20)

# create a bar chart for the top 10 items with width of 0.7
plt.bar(top_20_items["items"], top_20_items["incident_count"], width=0.9)
plt.xticks(rotation=90)
plt.xlabel("Items")
plt.ylabel("Incident Count")
plt.show()

import matplotlib.pyplot as plt

top_12_items = dF_table.sort_values("incident_count", ascending=False).head(12)

# specify the colors for the slices
colors = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a',
          '#ff7c43', '#ffa600', '#5390d9', '#7fb2f0', '#a7cced', '#d2e7f5']

plt.pie(top_12_items["incident_count"], labels=top_12_items["items"], autopct='%1.1f%%', colors=colors)
plt.show()

# importing required module
import plotly.express as px
# to have a same origin
dF_table["all"] = "all" 
# exclude the first row
dF_table = dF_table.iloc[1:]
# creating tree map using plotly
fig = px.treemap(dF_table.head(30), path=['all', "items"], values='incident_count',
                  color=dF_table["incident_count"].head(30), hover_data=['items'],
                  color_continuous_scale='Blues',
                )
# ploting the treemap
fig.show()

# replace all the NaN values with ‘’ and use inplace=True to commit the changes permanent into the dataframe
df.fillna('',axis=1,inplace=True)
df.head()

dF_table.tail(10)

# Transform Every Transaction to Seperate List & Gather Them into Numpy Array

transaction = []
for i in range(df.shape[0]):
    transaction.append([str(df.values[i,j]) for j in range(df.shape[1])])
    
transaction = np.array(transaction)
transaction

## Every trnasaction is in on list and combination of list is put in NP array

te = TransactionEncoder()
te_ary = te.fit(transaction).transform(transaction)
dataset = pd.DataFrame(te_ary, columns=te.columns_)
dataset

dataset.shape

first50 = dF_table["items"].head(50).values # Select Top50
dataset = dataset.loc[:,first50] # Extract Top50
dataset

dataset.columns
# We extracted first 50 column successfully.

## Items - All itesm are my cols 

## TRUE / FALSE  ->> 1 /0

def encode_units(x):
    if x == False:
        return 0 
    if x == True:
        return 1
    
dataset = dataset.applymap(encode_units)
dataset.head(10)

# Extracting the most frequest itemsets via Mlxtend
from mlxtend.frequent_patterns import apriori, association_rules
# min support - just to avoid the products which is very lesss occuring ( for 1-2 times in complete dataset )

frequent_itemsets = apriori(dataset, min_support=0.01, use_colnames=True) ## to generate itemset

# The length column has been added to increase ease of filtering.

frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets

frequent_itemsets[ (frequent_itemsets['length'] == 1) &
                   (frequent_itemsets['support'] >= 0.04)]

frequent_itemsets[ (frequent_itemsets['length'] == 2) ].head()

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules

rules["antecedents_length"] = rules["antecedents"].apply(lambda x: len(x))
rules["consequents_length"] = rules["consequents"].apply(lambda x: len(x))
rules.sort_values("lift",ascending=False)

# Sort values based on confidence

rules.sort_values("confidence",ascending=False)

rules[rules["antecedents"].str.contains("Montera-Lx", regex=False) & rules["antecedents_length"] == 0].sort_values("confidence", ascending=False).head(10)

rules[~rules["consequents"].str.contains("N95 mask", regex=False) & 
      ~rules["antecedents"].str.contains("N95 mask", regex=False)].sort_values("confidence", ascending=False).head(10)
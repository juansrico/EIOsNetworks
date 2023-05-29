"""
#### §§ This code is the step-by-step documentation of how I prepared the data downloaded from the YIO and conducted network analysis out of it §§    ####
STEPS:

0. Extract data from the online repository of the YIO
- Search criteria
  -> Environment & conservation. Categories: type I-> A,B,C,D,F. Type II (exclude): C,D,E,J
  -> Extraction led to 612 results
  -> Date: 07.02.2023
  
1. Data cleaning
2. Manual selection of organizations that do work in the environmental field
3. Transform dataset to a weighted matrix (edgelist) where the nodes are the countries and the ties are the number of shared memberships in EINGOs
4. Create network graphic and conduct analyses

"""
########### 2- Data cleaning ###############
# Load packages
import pandas as pd
import re


# Load data 
url = r"C:\Users\ricoe\OneDrive\Documents\WZB\PAPER\Original Data.csv"  
data = pd.read_csv(url)
Headers = ["Organisation", "members", "headquarters", "all", "timestamp"]
data.columns = Headers

# Extract aims from the column "all" and create the column "aims" of the organizations
def extract_aims(text):
    match = re.search("Aims\n.*\.", text)
    if match:
        aims = match.group()
        aims = re.sub("Aims\n", "", aims)
        return aims
    else:
        return None

data['aims'] = data['all'].apply(extract_aims)
print(data["aims"])

# Eliminate empty spaces in the column "organisation" 
remove_extra_spaces = lambda x: ' '.join(x.split()) # define lambda function to eliminate extra spaces
data['Organisation'] = data['Organisation'].apply(remove_extra_spaces) # apply lambda function to column
print(data["Organisation"]) # See results

# Extract member countries from the column "all" and create the column "members"
def extract_members(text):
    match = re.search("Member Countries.*?(?=Type)", text, re.DOTALL)
    if match:
        members = match.group()
        members = re.sub("Member Countries\n", "", members)
        members = members.strip()
        return members
    else:
        return None

data['members'] = data['all'].apply(extract_members) 
print(data["members"])

# Clean the new column "members" by replacing useless words with empty spaces
data['members'] = data['all'].apply(extract_members)
data['members'] = data['members'].str.replace('Member Countries & Regions', '')

# Create a column to differentiate IOs from INGOs
def contains_specific_words(text):
    if ("Type II Classificationg: intergovernmental organization" in text) or ("Type II Classificationg" in text):
        return 1
    else:
        return 0

data['IO'] = data['all'].apply(contains_specific_words)
print(data["IO"].head(50))

# Move column "aims" to the second position and shift all other columns to the right
cols = data.columns.tolist()
cols = cols[:1] + [cols[5]] + cols[1:5] + cols[6:]
data = data[cols]

# Save to CSV
data.to_csv("DataProcessing1", index=False)

########### 3- Manual selection ###############
"""
1 - Delete organizations whose aims are not related to environmental protection or fight against climate change, or that do not have membership data: 
All those deleted case were listed in the document "deleted organizations final".
(QUESTION: What is the sheet "orgas with unfitting goals" (col) doing?

2- Clean the column memberships: 
Deleted words such as "individuals in xxx countries" and included all countries listed, independent of whether it said "partner organizations in xxx countries" 
or something similar

3 - Delete all non-sovereign countries listed:
Anguilla, St Eustatius, Faeroe Is, Macau, Montserrat, St Helena, Gibraltar, Virgin Is USA, Samoa USA, Norfolk Is, St Maarten, Cayman Is, New Caledonia, Northern Mariana Is, Virgin Is UK  Neth Antilles (dissolved!)

4- Replace all cities listed with the countries’ name:
Aguadilla & Mayagüez: Puerto Rico
Dar es Salaam: Tanzania

5- Save document as "xxx".

"""

######## 4 - Create matrix for the network analysis  ###################

# Before that: delete double listings of the same country in a specific space: 
def remove_duplicates(row): # Define a function to remove repeated words within a string
    words = row.split(', ')
    return ', '.join(sorted(set(words), key=words.index))

df['members'] = df['members'].apply(remove_duplicates) # Apply the function to the members column
print(df["members"])

#Create matrix
from itertools import combinations

# Assuming your DataFrame is called 'df'
countries = set(df['members'].str.split(', ').sum())
matrix = pd.DataFrame(0, index=countries, columns=countries)

# Count the occurrences of each pair of countries
for row in df['members']:
    for a, b in combinations(row.split(', '), 2):
        matrix.loc[a, b] += 1
        matrix.loc[b, a] += 1

# Print the resulting matrix
print(matrix)

dd = pd.DataFrame(matrix)

# save dataframe as a csv file
dd.to_csv('mydata.csv', index=True)


######## START THE ANALYSIS ############## 
# This procedure takes the file "mydata.csv" (edgelist format) and makes a network out of it.

# Load the data
with open(r'C:\Users\ricoe\OneDrive\Documents\WZB\PAPER\mydata.csv', 'r') as f:
    reader = csv.reader(f)
    node_names = next(reader)[1:]  # extract the node names from the first row
    weighted_matrix = np.array([[float(x) for x in row[1:]] for row in reader])

# Create the graph (network)
G = nx.Graph()

# Add nodes to the graph
num_nodes = len(node_names)
G.add_nodes_from(node_names)

# Add weighted edges to the graph
for i in range(num_nodes):
    for j in range(num_nodes):
        weight = weighted_matrix[i][j]
        if weight != 0:
            G.add_edge(node_names[i], node_names[j], weight=weight)


            
            
 
import csv
import numpy as np
import networkx as nx


# Load the data
with open(r'C:\Users\ricoe\OneDrive\Documents\WZB\PAPER\mydata.csv', 'r') as f:
    reader = csv.reader(f)
    node_names = next(reader)[1:]  # extract the node names from the first row
    weighted_matrix = np.array([[float(x) for x in row[1:]] for row in reader])

# Create the graph
G = nx.Graph()

# Add nodes to the graph
num_nodes = len(node_names)
G.add_nodes_from(node_names)

# Add weighted edges to the graph
for i in range(num_nodes):
    for j in range(num_nodes):
        weight = weighted_matrix[i][j]
        if weight != 0:
            G.add_edge(node_names[i], node_names[j], weight=weight)

            
### PRINT NETWORKS' METRICS            
print(nx.info(G))  # Get basic info about the network

density = nx.density(G)
print("Network density:", density)

num_nodes = nx.number_of_nodes(G)
num_edges = nx.number_of_edges(G)
density = nx.density(G)
avg_clustering_coef = nx.average_clustering(G)
avg_shortest_path_len = nx.average_shortest_path_length(G)
degree_centralities = nx.degree_centrality(G)
closeness_centralities = nx.closeness_centrality(G)
betweenness_centralities = nx.betweenness_centrality(G)

print("Number of nodes:", num_nodes)
print("Number of edges:", num_edges)
print("Density:", density)
print("Average clustering coefficient:", avg_clustering_coef)
print("Average shortest path length:", avg_shortest_path_len)
print("Degree centralities:", degree_centralities)
print("Closeness centralities:", closeness_centralities)
print("Betweenness centralities:", betweenness_centralities)

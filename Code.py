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
4. Create metrics & conduct analyses
5. Create graph

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
# This procedure takes the file "xxx.csv" (edgelist format) and makes a network out of it.

#import csv
import numpy as np
import networkx as nx
import pandas as pd

# Step 1: Load the data and create an undirected network
file_path = 'C:/Users/ricoe/OneDrive/Documents/WZB/PAPER/FINAL DATASETS/Archive/Matrix network Final _ Scotland and Tanzania2 deleted.csv'
df = pd.read_csv(file_path, sep=';')
node_names = df.columns[1:].tolist()
weighted_matrix = df.iloc[:, 1:].to_numpy()

G = nx.Graph()
for i, node in enumerate(node_names):
    for j in range(i+1, len(node_names)):
        weight = weighted_matrix[i, j]
        if weight != 0:
            G.add_edge(node_names[i], node_names[j], weight=weight)

# Step 2: Print network metrics
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
density = nx.density(G)
avg_clustering_coef = nx.average_clustering(G)
avg_shortest_path_len = nx.average_shortest_path_length(G) if nx.is_connected(G) else "N/A (Graph is not connected)"

print("Network metrics:")
print(f"Number of nodes: {num_nodes} (The total number of nodes in the network)")
print(f"Number of edges: {num_edges} (The total number of edges in the network)")
print(f"Network density: {density} (A measure of how dense the network is, ranging from 0 to 1)")
print(f"Average clustering coefficient: {avg_clustering_coef} (The average local clustering coefficient of the nodes)")
print(f"Average shortest path length: {avg_shortest_path_len} (The average shortest path length between all pairs of nodes)")

# Step 3: Calculate individual centrality measures
centrality_measures = {
    'Node': node_names,
    'Degree Centrality': [nx.degree_centrality(G)[node] for node in node_names],
    'Closeness Centrality': [nx.closeness_centrality(G)[node] for node in node_names],
    'Betweenness Centrality': [nx.betweenness_centrality(G)[node] for node in node_names],
    'Eigenvector Centrality': [nx.eigenvector_centrality(G, max_iter=1000)[node] for node in node_names],
    'Clustering Coefficient': [nx.clustering(G)[node] for node in node_names]
}

# Step 4: Save the centrality measures to a CSV file
output_file_path = r"C:\Users\ricoe\OneDrive\Documents\WZB\PAPER\FINAL DATASETS\Metrics network analysis FINAL FINAAAAAl.csv"  # Adjust the file path as needed
centrality_df = pd.DataFrame(centrality_measures)
centrality_df.to_csv(output_file_path, index=False)

print("Centrality measures saved to:", output_file_path)


# (Optional) # Create correlation matrix and heatmap for facilitating the interpretation 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Replace the file path with the actual path to your CSV file
file_path = r"C:\Users\ricoe\OneDrive\Documents\WZB\PAPER\FINAL DATASETS\Matrix network Final.csv"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path, index_col=0)

# Calculate correlation matrix to identify patterns in shared memberships
correlation_matrix = df.corr()

# Export the correlation matrix to a CSV file
correlation_matrix.to_csv(r"C:\Users\ricoe\OneDrive\Documents\WZB\PAPER\FINAL DATASETS\CorrelationMatrix.csv")

# Visualize the correlation matrix using a heatmap with clear titles
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Shared Memberships")
plt.xlabel("Countries")
plt.ylabel("Countries")
plt.savefig(r"C:\Users\ricoe\OneDrive\Documents\WZB\PAPER\FINAL DATASETS\CorrelationMatrixHeatmap.png")
plt.show()

print("Correlation Matrix exported to CorrelationMatrix.csv")
print("Heatmap saved as CorrelationMatrixHeatmap.png")


###### 5- Create Graphic #########

### printing the graph

import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load the data
with open(r'C:\Users\ricoe\OneDrive\Documents\WZB\PAPER\mydata4.csv', 'r') as f:
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

# Compute node centrality
centrality = nx.degree_centrality(G)

# Filter extreme outliers (optional)
threshold = 0.1  # Adjust the threshold as needed
filtered_nodes = [node for node, centrality_value in centrality.items() if centrality_value > threshold]
G_filtered = G.subgraph(filtered_nodes)

# Compute node sizes based on centrality
node_sizes = [centrality[node] * 2000 for node in G_filtered.nodes()]

# Compute node colors based on centrality
centrality_values = [centrality[node] for node in G_filtered.nodes()]
min_centrality = min(centrality_values)
max_centrality = max(centrality_values)
norm_centrality_values = [(np.log10(x) - np.log10(min_centrality)) / (np.log10(max_centrality) - np.log10(min_centrality)) for x in centrality_values]
color_map = cm.ScalarMappable(cmap=cm.viridis)
color_map.set_array(centrality_values)
node_colors = color_map.to_rgba(norm_centrality_values)

# Visualize the network
pos = nx.spring_layout(G_filtered, seed=42)  # layout algorithm for node positioning
plt.figure(figsize=(40, 40))  # adjust the figure size
nx.draw_networkx(G_filtered, pos=pos, with_labels=True, node_color=node_colors, edge_color='gray', alpha=0.8,
                 node_size=node_sizes, font_size=8, font_color='black', font_weight='bold')

# Save the network visualization to a file
plt.savefig(r'C:\Users\ricoe\OneDrive\Documents\WZB\PAPER\network_visualization2.jpg')

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






-----------------------------------------------------------
### CREATING CLIMATE AMBITION INDEX ##########

import pandas as pd

# Replace 'your_file_path.csv' with the path to your dataset
file_path = r"C:\Users\ricoe\OneDrive\Documents\WZB\PAPER\FINAL DATASETS\Climate Ambition Index\Climate Ambition Scores.csv"

# Load the dataset
df = pd.read_csv(file_path)

# Function to calculate the 'endgoal' component of the index
def calculate_endgoal(row):
    return (row['end_target'] * 0.30 + row['end_target_year'] * 0.30 + row['end_target_status'] * 0.40) * 0.40

# Function to calculate the 'interim' component of the index
def calculate_interim(row):
    return (row['interim_target'] * 0.40 + row['interim_target_percentage_reduction'] * 0.60) * 0.40

# Calculate the 'endgoal' component for all rows
df['endgoal'] = df.apply(calculate_endgoal, axis=1)

# Calculate the 'interim' component for all rows
df['interim'] = df.apply(calculate_interim, axis=1)

# Calculate the 'transparency' component for all rows
df['transparency'] = df['reporting_mechanism'] * 0.20

# Sum the components to get the final climate ambition index
df['climate_ambition_index'] = df['endgoal'] + df['interim'] + df['transparency']

# Normalize the climate ambition index
min_score = df['climate_ambition_index'].min()
max_score = df['climate_ambition_index'].max()
df['normalized_climate_ambition_index'] = (df['climate_ambition_index'] - min_score) / (max_score - min_score)

# Optionally, drop the intermediate calculation columns
df.drop(['endgoal', 'interim', 'transparency'], axis=1, inplace=True)

# Save the updated DataFrame to a new CSV file
# Replace 'updated_file_path.csv' with your desired output file path
output_file_path = r"C:\Users\ricoe\OneDrive\Documents\WZB\PAPER\FINAL DATASETS\THIS IS.csv"
df.to_csv(output_file_path, index=False)


-----------------------------
### CREATING THE NETWORK GRAPHS (this one exemplary for world regions)

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Adjust the file path as necessary for your environment
file_path = "/content/Network Matrix Final March 2024.csv"

# Reading the CSV file, assuming the first column should be used as the index
df = pd.read_csv(file_path, delimiter=';', index_col=0)

# Ensure the columns match the index to represent nodes correctly
df.columns = df.index.tolist()

# Create a graph from the adjacency matrix
G = nx.from_pandas_adjacency(df, create_using=nx.Graph())

# Prepare 'world_regions' with your country-to-region mapping
# Ensure all countries are included or they will be mapped to "Undefined"
world_regions = {
    "Afghanistan": "South Asia",
    "Andorra": "Europe and Central Asia",
    "Albania": "Europe and Central Asia",
    "Algeria": "Middle East and North Africa",
    "Angola": "Sub-Saharan Africa",
    "Antigua-Barbuda": "Latin America and Caribbean",
    "Argentina": "Latin America and Caribbean",
    "Armenia": "Europe and Central Asia",
    "Aruba": "Latin America and Caribbean",
    "Australia": "East Asia and Pacific",
    "Austria": "Europe and Central Asia",
    "Azerbaijan": "Europe and Central Asia",
    "Bahamas": "Latin America and Caribbean",
    "Bahrain": "Middle East and North Africa",
    "Bangladesh": "South Asia",
    "Barbados": "Latin America and Caribbean",
    "Belarus": "Europe and Central Asia",
    "Belgium": "Europe and Central Asia",
    "Belize": "Latin America and Caribbean",
    "Benin": "Sub-Saharan Africa",
    "Bhutan": "South Asia",
    "Bolivia": "Latin America and Caribbean",
    "Bosnia-Herzegovina": "Europe and Central Asia",
    "Botswana": "Sub-Saharan Africa",
    "Brazil": "Latin America and Caribbean",
    "Bulgaria": "Europe and Central Asia",
    "Burkina Faso": "Sub-Saharan Africa",
    "Brunei Darussalam": "East Asia and Pacific",
    "Burundi": "Sub-Saharan Africa",
    "Cambodia": "East Asia and Pacific",
    "Cameroon": "Sub-Saharan Africa",
    "Canada": "North America",
    "Cape Verde": "Sub-Saharan Africa",
    "Central African Rep": "Sub-Saharan Africa",
    "Chad": "Sub-Saharan Africa",
    "Chile": "Latin America and Caribbean",
    "China": "East Asia and Pacific",
    "Colombia": "Latin America and Caribbean",
    "Comoros": "Sub-Saharan Africa",
    "Congo Brazzaville": "Sub-Saharan Africa",
    "Congo DR": "Sub-Saharan Africa",
    "Costa Rica": "Latin America and Caribbean",
    "Cote d Ivoire": "Sub-Saharan Africa",
    "Croatia": "Europe and Central Asia",
    "Cuba": "Latin America and Caribbean",
    "Curaçao": "Latin America and Caribbean",
    "Côte d'Ivoire": "Sub-Saharan Africa",
    "Cyprus": "Europe and Central Asia",
    "Czechia": "Europe and Central Asia",
    "Denmark": "Europe and Central Asia",
    "Djibouti": "Middle East and North Africa",
    "Dominica": "Latin America and Caribbean",
    "Dominican Rep": "Latin America and Caribbean",
    "Ecuador": "Latin America and Caribbean",
    "Egypt": "Middle East and North Africa",
    "El Salvador": "Latin America and Caribbean",
    "Equatorial Guinea": "Sub-Saharan Africa",
    "Eritrea": "Sub-Saharan Africa",
    "Estonia": "Europe and Central Asia",
    "Eswatini": "Sub-Saharan Africa",
    "Ethiopia": "Sub-Saharan Africa",
    "Fiji": "East Asia and Pacific",
    "Finland": "Europe and Central Asia",
    "France": "Europe and Central Asia",
    "Gabon": "Sub-Saharan Africa",
    "Gambia": "Sub-Saharan Africa",
    "Georgia": "Europe and Central Asia",
    "Germany": "Europe and Central Asia",
    "Ghana": "Sub-Saharan Africa",
    "Greece": "Europe and Central Asia",
    "Grenada": "Latin America and Caribbean",
    "Guatemala": "Latin America and Caribbean",
    "Guinea": "Sub-Saharan Africa",
    "Guinea-Bissau": "Sub-Saharan Africa",
    "Guyana": "Latin America and Caribbean",
    "Haiti": "Latin America and Caribbean",
    "Honduras": "Latin America and Caribbean",
    "Hungary": "Europe and Central Asia",
    "Iceland": "Europe and Central Asia",
    "India": "South Asia",
    "Indonesia": "East Asia and Pacific",
    "Iran": "Middle East and North Africa",
    "Iraq": "Middle East and North Africa",
    "Ireland": "Europe and Central Asia",
    "Israel": "Middle East and North Africa",
    "Italy": "Europe and Central Asia",
    "Jamaica": "Latin America and Caribbean",
    "Japan": "East Asia and Pacific",
    "Jordan": "Middle East and North Africa",
    "Kazakhstan": "Europe and Central Asia",
    "Kenya": "Sub-Saharan Africa",
    "Kiribati": "East Asia and Pacific",
    "Korea Rep": "East Asia and Pacific",
    "Kuwait": "Middle East and North Africa",
    "Kyrgyzstan": "Europe and Central Asia",
    "Laos": "East Asia and Pacific",
    "Latvia": "Europe and Central Asia",
    "Libya": "Middle East and North Africa",
    "Lebanon": "Middle East and North Africa",
    "Lesotho": "Sub-Saharan Africa",
    "Liberia": "Sub-Saharan Africa",
    "Lithuania": "Europe and Central Asia",
    "Liechtenstein": "Europe and Central Asia",
    "Luxembourg": "Europe and Central Asia",
    "Madagascar": "Sub-Saharan Africa",
    "Malawi": "Sub-Saharan Africa",
    "Malaysia": "East Asia and Pacific",
    "Maldives": "South Asia",
    "Mali": "Sub-Saharan Africa",
    "Malta": "Europe and Central Asia",
    "Mauritania": "Sub-Saharan Africa",
    "Mauritius": "Sub-Saharan Africa",
    "Mexico": "Latin America and Caribbean",
    "Micronesia FS": "East Asia and Pacific",
    "Moldova": "Europe and Central Asia",
    "Mongolia": "East Asia and Pacific",
    "Montenegro": "Europe and Central Asia",
    "Morocco": "Middle East and North Africa",
    "Mozambique": "Sub-Saharan Africa",
    "Myanmar": "East Asia and Pacific",
    "Namibia": "Sub-Saharan Africa",
    "Nauru": "East Asia and Pacific",
    "Nepal": "South Asia",
    "Netherlands": "Europe and Central Asia",
    "New Zealand": "East Asia and Pacific",
    "Nicaragua": "Latin America and Caribbean",
    "Niger": "Sub-Saharan Africa",
    "Nigeria": "Sub-Saharan Africa",
    "North Macedonia": "Europe and Central Asia",
    "Norway": "Europe and Central Asia",
    "Oman": "Middle East and North Africa",
    "Pakistan": "South Asia",
    "Palau": "East Asia and Pacific",
    "Palestine": "Middle East and North Africa",
    "Panama": "Latin America and Caribbean",
    "Papua New Guinea": "East Asia and Pacific",
    "Paraguay": "Latin America and Caribbean",
    "Peru": "Latin America and Caribbean",
    "Philippines": "East Asia and Pacific",
    "Poland": "Europe and Central Asia",
    "Portugal": "Europe and Central Asia",
    "Qatar": "Middle East and North Africa",
    "Romania": "Europe and Central Asia",
    "Russia": "Europe and Central Asia",
    "Rwanda": "Sub-Saharan Africa",
    "Sao Tome Principe": "Sub-Saharan Africa",
    "Saudi Arabia": "Middle East and North Africa",
    "Senegal": "Sub-Saharan Africa",
    "Serbia": "Europe and Central Asia",
    "Seychelles": "Sub-Saharan Africa",
    "Sierra Leone": "Sub-Saharan Africa",
    "Singapore": "East Asia and Pacific",
    "Slovakia": "Europe and Central Asia",
    "Slovenia": "Europe and Central Asia",
    "Solomon Is": "East Asia and Pacific",
    "Somalia": "Sub-Saharan Africa",
    "South Africa": "Sub-Saharan Africa",
    "Spain": "Europe and Central Asia",
    "Sri Lanka": "South Asia",
    "St Lucia": "Latin America and Caribbean",
    "Sudan": "Sub-Saharan Africa",
    "Suriname": "Latin America and Caribbean",
    "Sweden": "Europe and Central Asia",
    "Switzerland": "Europe and Central Asia",
    "Syrian AR": "Middle East and North Africa",
    "Tajikistan": "Europe and Central Asia",
    "Taiwan": "East Asia and Pacific",
    "Tanzania": "Sub-Saharan Africa",
    "Thailand": "East Asia and Pacific",
    "Timor Leste": "East Asia and Pacific",
    "Togo": "Sub-Saharan Africa",
    "Tonga": "East Asia and Pacific",
    "Trinidad-Tobago": "Latin America and Caribbean",
    "Tunisia": "Middle East and North Africa",
    "Turkey": "Europe and Central Asia",
    "Turkmenistan": "Europe and Central Asia",
    "Tuvalu": "East Asia and Pacific",
    "Uganda": "Sub-Saharan Africa",
    "Ukraine": "Europe and Central Asia",
    "United Arab Emirates": "Middle East and North Africa",
    "Uruguay": "Latin America and Caribbean",
    "USA": "North America",
    "Uzbekistan": "Europe and Central Asia",
    "Vanuatu": "East Asia and Pacific",
    "Venezuela": "Latin America and Caribbean",
    "Vietnam": "East Asia and Pacific",
    "Yemen": "Middle East and North Africa",
    "Zambia": "Sub-Saharan Africa",
    "Marshall Is": "East Asia and Pacific",
    "Timor-Leste": "East Asia and Pacific",
    "Monaco": "Europe and Central Asia",
    "Bermuda": "North America",
    "St Kitts-Nevis": "Latin America and Caribbean",
    "Sao Tomé-Principe": "Sub-Saharan Africa",
    "South Sudan": "Sub-Saharan Africa",
    "St Vincent-Grenadines": "Latin America and Caribbean",
    "Kosovo": "Europe and Central Asia",
    "San Marino": "Europe and Central Asia",
    "Niue": "East Asia and Pacific",
    "Cook Is": "East Asia and Pacific",
    "Korea DPR": "East Asia and Pacific",
    "St Maarten": "Latin America and Caribbean",
    "Iran Islamic Rep": "Middle East and North Africa",
    "UK": "Europe and Central Asia",
    "Türkiye": "Europe and Central Asia",
    "Tanzania UR": "Sub-Saharan Africa",
    "Zimbabwe": "Sub-Saharan Africa"
}

# Define colors for each world region, including 'Undefined'
region_colors = {
    "South Asia": "#4169E1",  # Royal Blue
    "Europe and Central Asia": "#DC143C",  # Crimson
    "Middle East and North Africa": "#228B22",  # Forest Green
    "Sub-Saharan Africa": "#DAA520",  # Goldenrod
    "Latin America and Caribbean": "#BA55D3",  # Medium Orchid
    "East Asia and Pacific": "#008080",  # Teal
    "North America": "#A0522D",  # Sienna
    "Undefined": "#808080"  # Gray for undefined regions
}

# Calculate degree centrality for each node
centrality = nx.degree_centrality(G)
# Scale centrality values for node size visualization (adjust the scaling factor as needed)
node_sizes = [v * 100 for v in centrality.values()]

# Assign color to each node based on its region
node_colors = [region_colors[world_regions[node]] if node in world_regions else region_colors["Undefined"] for node in G.nodes()]

# Adjusting the spring layout's parameters to optimize distances between nodes
plt.figure(figsize=(20, 20))
pos = nx.spring_layout(G, k=0.6/(np.sqrt(G.order())), iterations=50)

# Draw nodes with sizes scaled by centrality and specified color
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.7)

# Draw edges with thinner lines for clarity
nx.draw_networkx_edges(G, pos, edge_color='gray', width=0.03, alpha=0.5)
# Position labels close to nodes to minimize overlap, adjusting font size for readability
nx.draw_networkx_labels(G, pos, font_size=6, alpha=0.9)

# Adding legend for world regions
legend_labels = list(set(world_regions.values()))
patches = [plt.Line2D([0], [0], marker='o', color='w', label=region,
                      markerfacecolor=color, markersize=10) for region, color in region_colors.items() if region in legend_labels]
plt.legend(handles=patches, title="World Regions", loc='best', fontsize='small')

plt.title("Network of Shared Memberships in International Environmental Organizations", fontsize=16)
plt.axis('off')
plt.tight_layout()

# To display the plot
plt.show()

# Optionally, save the plot to a file
plt.savefig('network_visualization_centrality.png', format='png', dpi=300, bbox_inches='tight')

-------------------------
### Running the regressions

import math
import numpy as np
import pandas as pd
import statistics
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols
from scipy.stats import shapiro, probplot
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.gofplots import qqplot
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import os
from docx import Document
from docx import Document
from docx.shared import Inches
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib.pyplot as plt
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro

### First check the assumptions for regression
#Multiple regression 
# Replace 'inf' values with 'NaN'
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with 'NaN' in any column (or consider imputation)
df.dropna(inplace=True)

# Define your dependent variable and independent variables
dependent_var = 'Ambition'
independent_vars = ['Embededdness score', 'GDP per capita lg', 'Democracy', "Population 2022 log"] 

# Prepare the data for the model
X = df[independent_vars]
y = df[dependent_var]

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the multiple regression model
model = sm.OLS(y, X).fit()

# Print out the model's summary
print(model.summary())
------------------------------
#### REAL REGRESSION

# Load the dataset
df = pd.read_excel(r"C:\Users\ricoe\OneDrive\Documents\WZB\PAPER\FINAL DATASETS\Final Final Final datasets\FULL DATASET 25.02.xlsx")

## Logarithm of variables
df['GDP per capita lg'] = np.log(df['GDP per capita'])
df['Population 2022 lg'] = np.log(df['Population 2022'])
df['Ambition lg'] = np.log(df['Ambition'])
df['GDP per capita lg'] = np.log(df['GDP per capita'])
df['Eigenvector lg'] = np.log(df['Eigenvector'])
df['CO2 2022 lg'] = np.log(df['CO2 2022'])
df['EPI 2022 lg'] = np.log(df['EPI 2022'])
df['Democracy lg'] = np.log(df['Democracy'])

#Multiple regression 
# Replace 'inf' values with 'NaN'
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with 'NaN' in any column (or consider imputation)
df.dropna(inplace=True)

# Define your dependent variable and independent variables
dependent_var = 'Ambition'
independent_vars = ['Embededdness score', 'GDP per capita lg', 'Democracy', "Population 2022 log"] 

# Prepare the data for the model
X = df[independent_vars]
y = df[dependent_var]

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the multiple regression model
model = sm.OLS(y, X).fit()

# Print out the model's summary
print(model.summary())



------------------
### Making scatter plots
# MAKING THE SCATTER PLOTS

import matplotlib.pyplot as plt
import seaborn as sns

# Set the style of the plot to make it more appealing
sns.set_style("darkgrid")

# Create the scatter plot with more appealing aesthetics
plt.figure(figsize=(10, 6))  # Set the figure size
plt.scatter(df['Embededdness score'], df['EPI 2022'], alpha=0.7, edgecolors='w', s=100)  # Assume 'df' is your DataFrame

# Set the title and labels with a specified font
title_font = {'fontname':'Georgia', 'size':'11'}
axis_font = {'fontname':'Georgia', 'size':'11'}

plt.title('Relationship between Degree of Embededdness and Environmental Performance Index', **title_font)
plt.xlabel('Embededdness score', **axis_font)
plt.ylabel('Environmental Performance Index', **axis_font)

# Set x-axis to only show levels 1 through 5
plt.xticks([1, 2, 3, 4, 5], **axis_font)

# Optional: If you want to remove the top and right axis spines but keep the grid
sns.despine()

# Save the plot with a transparent background to the specified directory
plt.savefig(r'C:\Users\ricoe\OneDrive\Documents\WZB\PAPER\FINAL DATASETS\Archive\scatter_plot2.png', bbox_inches='tight', transparent=True)

# Show the plot
plt.show()

-----------------------------
## Making the histogramms

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Set the style of the plot to make it more appealing
sns.set_style("darkgrid")

# Create the histogram with more appealing aesthetics
plt.figure(figsize=(10, 6))  # Set the figure size
sns.histplot(df['Ambition'], kde=False, color=(75/255, 0/255, 130/255), bins=30)  # RGB values must be between 0 and 1 for matplotlib

# Set the title and labels with a specified font
title_font = {'fontname':'Georgia', 'size':'11'}
axis_font = {'fontname':'Georgia', 'size':'11'}

plt.title('Histogram of the Climate Ambition Index', **title_font)
plt.xlabel('Climate Ambition Index', **axis_font)
plt.ylabel('Frequency', **axis_font)

# Optional: If you want to remove the top and right axis spines but keep the grid
sns.despine()

# Save the plot with a transparent background
plt.savefig(r"C:\Users\ricoe\OneDrive\Documents\WZB\PAPER\FINAL DATASETS\Archive\histogramm.png", bbox_inches='tight', transparent=True)

# Show the plot
plt.show()


---------------------
###Correlation matrix
import pandas as pd
import numpy as np

# Assuming 'df' is your DataFrame

# Define the list of specific variables/columns to include
columns_of_interest = [
    'Eigenvector',
    'EPI 2022',
    'GDP per capita log',
    'Democracy',
    'Population 2022 log',
    'Ambition',
    'Embededdness score'
]

# Create a subset of your DataFrame with only the specified columns
numeric_df = df[columns_of_interest]

# Calculate the correlation matrix for the subset of columns
correlation_matrix = numeric_df.corr()

# Convert the correlation matrix to a DataFrame for easier manipulation
correlation_df = pd.DataFrame(correlation_matrix)

# Reset index to turn the index into a column, useful for table conversion
correlation_df_reset = correlation_df.reset_index()

# Export the DataFrame to an Excel file
correlation_df_reset.to_excel(r"C:\Users\ricoe\OneDrive\Documents\WZB\PAPER\FINAL DATASETS\Archive\correlation.xlsx", index=False)

print("Correlation matrix exported successfully.")

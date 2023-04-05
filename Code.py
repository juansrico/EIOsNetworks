# Data cleaning
import pandas as pd
url = r"C:\Users\ricoe\OneDrive\Documents\WZB\PAPER\Original Data.csv"
data = pd.read_csv(url)
Headers = ["Organisation", "members", "headquarters", "all", "timestamp"]
data.columns = Headers


# Extract aims from the column "all"
import re
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

# define lambda function to eliminate extra spaces
remove_extra_spaces = lambda x: ' '.join(x.split())

# apply lambda function to column
data['Organisation'] = data['Organisation'].apply(remove_extra_spaces)
# See results
print(data["Organisation"])

# Extract member countries from the column "all"
import re
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


# Manual process: delete organizations whose aims dont fit or dont have membership data (All those deleted case were listed in the document "deleted organizations".#
# An additional manual step was to clean the column memberships (I deleted words such as "individuals in xxx countries" and included all countries listed, independent of whether it said "partner organizations in xxx countries" or similar

### Create matrix for the network analysis

# Before that: delete double listings of the same country in a specific space: 


# Assuming your DataFrame is called 'df' and 'members' is the column with countries separated by commas
df['members'] = df['members'].apply(lambda x: ', '.join(sorted(set(x.split(', ')))))

df.to_csv("DataProcessing6", index=False)



##Create matrix

import pandas as pd
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



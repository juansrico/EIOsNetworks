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


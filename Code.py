# Data cleaning
import pandas as pd
url = r"C:\Users\ricoe\OneDrive\Documents\WZB\PAPER\Fulldata current.csv"
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

# Save to CSV
data.to_csv("sample_df.csv", index=False)


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

data.to_csv("Fulldata Current2", index=False)

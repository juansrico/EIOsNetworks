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

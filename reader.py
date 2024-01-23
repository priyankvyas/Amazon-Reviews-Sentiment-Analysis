import gzip
import json
import pandas as pd
import numpy as np

# Parse and load the JSON object
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)

# Build the Pandas DataFrame from the JSON object
def get_DF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

# Return the Pandas DataFrame after filtering out the irrelevant columns
# and mapping the numeric score to corresponding categorical values
def load_DF(path):
    df = get_DF(path)
    df = keep_columns(df, ["overall", "reviewText", "summary"])
    df["label"] = np.floor(df["overall"] / 4)
    return df

# Return a Pandas DataFrame with only the desired columns retained
def keep_columns(df, columns):
    df = df.drop(df.columns.difference(columns), axis=1)
    return df
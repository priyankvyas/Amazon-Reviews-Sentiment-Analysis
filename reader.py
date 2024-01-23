import gzip
import json
import pandas as pd

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)

def get_DF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def load_DF(path):
    df = get_DF(path)
    cols_to_keep = ["overall", "reviewText", "summary"]
    df = df.drop(df.columns.difference(cols_to_keep), axis=1)
    labels_map = {
        1.0: '0',
        2.0: '0',
        3.0: '0',
        4.0: '1',
        5.0: '1'
    }
    df["label"] = df["overall"].map(labels_map)
    return df
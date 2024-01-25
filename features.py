import numpy as np
from collections import defaultdict

# Build a frequency dictionary with all the words in the vocabulary
def build_freq_dict(df):
    freq_dict = defaultdict(int)
    for _, row in df.iterrows():
        for word in row["preprocessedText"].split(' '):
            freq_dict[(word, row["label"])] += 1
    return freq_dict

# Get the document vectors for each row of the Pandas DataFrame
def get_vectors(df, freq_dict):
    vectors = np.zeros((len(df.index), 3))
    for index, row in df.iterrows():
        vectors[index] = get_doc_vector(row["preprocessedText"], freq_dict)
    return vectors

# Compose the document vector for a given piece of text based on the frequency
# of the words' occurrence in texts of each label
def get_doc_vector(text, freq_dict):
    vector = np.array([1, 0, 0])
    for word in text.split(' '):
        vector[1] += freq_dict[(word, '0')]
        vector[2] += freq_dict[(word, '1')]
    return vector
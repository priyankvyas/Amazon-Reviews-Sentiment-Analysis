import preprocess
import numpy as np
from nltk.corpus import stopwords
from collections import defaultdict

# Build a vocabulary of distinct words found in the text corpus
def build_vocab(df):
    vocab = set()
    for index, row in df.iterrows():
        df.loc[index, "preprocessedText"] = preprocess.preprocess(str(row["reviewText"]) + " " + str(row["summary"]))
        vocab.update(df.loc[index, "preprocessedText"].split(' '))

    vocab.discard('')
    vocab = vocab - set(stopwords.words('english'))
    return vocab

def build_freq_dict(df, vocab):
    freq_dict = defaultdict(int)
    for index, row in df.iterrows():
        update_freq(row["preprocessedText"], row["label"], vocab, freq_dict)
    return freq_dict

def update_freq(text, label, vocab, freq_dict):
    for word in text.split(' '):
        if word in vocab:
            freq_dict[(word, label)] += 1

def get_vectors(df, freq_dict):
    vectors = np.zeros((len(df.index), 3))
    for index, row in df.iterrows():
        vectors[index] = get_doc_vector(row["preprocessedText"], freq_dict)
    return vectors

def get_doc_vector(text, freq_dict):
    vector = np.array([1, 0, 0])
    for word in text.split(' '):
        vector[1] += freq_dict[(word, '0')]
        vector[2] += freq_dict[(word, '1')]
    return vector
import reader
import features
import logistic_regression

# First, load the raw data into a form that 
df = reader.load_DF('All_Beauty_5.json.gz')
vocab, labels = features.build_vocab(df)
freq_dict = features.build_freq_dict(df, vocab)
vectors = features.get_vectors(df)

train_X = vectors[:4000, :]
train_Y = labels[:4000]
test_X = vectors[4000:, :]
test_Y = labels[4000:]

logistic_regression.gradient_descent(train_X, train_Y)
import reader
import features
import logistic_regression

# First, load the raw data into a Pandas DataFrame
df = reader.load_DF('All_Beauty_5.json.gz')

# Build a vocabulary of all the observed words in the data
vocab = features.build_vocab(df)

# Build a frequency dictionary of all the words in the vocabulary
freq_dict = features.build_freq_dict(df, vocab)

# Get the document vectors for the reviews i.e., the title and the summary
vectors = features.get_vectors(df, freq_dict)

# Get the list of labels from the Pandas DataFrame
labels = df["label"].to_numpy()

# Split the data into training and test sets
train_X = vectors[:4000, :]
train_Y = labels[:4000]
test_X = vectors[4000:, :]
test_Y = labels[4000:]

# Train the logistic regression model using gradient descent
logistic_regression.gradient_descent(train_X, train_Y)
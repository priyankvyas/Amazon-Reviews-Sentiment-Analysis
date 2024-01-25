import reader
import preprocess
import features
import logistic_regression

# First, load the raw data into a Pandas DataFrame
df = reader.load_DF('All_Beauty_5.json.gz')

# Preprocess the review text in the Pandas DataFrame
preprocessedDf = preprocess.preprocess(df)

# Build a frequency dictionary of all the words in the vocabulary
freq_dict = features.build_freq_dict(preprocessedDf)

# Get the document vectors for the reviews i.e., the title and the summary
vectors = features.get_vectors(preprocessedDf, freq_dict)

# Get the list of labels from the Pandas DataFrame
labels = preprocessedDf["label"].to_numpy()

# Split the data into training and validation sets
train_X = vectors[:4000, :]
train_Y = labels[:4000]
validation_X = vectors[4000:, :]
validation_Y = labels[4000:]

# Train the logistic regression model using gradient descent
logistic_regression.gradient_descent(train_X, train_Y)
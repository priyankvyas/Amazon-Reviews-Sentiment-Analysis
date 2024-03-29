import reader
import preprocess
import features
import logistic_regression
import math

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
train_X = vectors[: math.floor(df.shape[0] * 0.8), :]
train_Y = labels[: len(train_X)]
validation_X = vectors[math.floor(df.shape[0] * 0.8):, :]
validation_Y = labels[len(train_X):]

# Train the logistic regression model using gradient descent
model_weights = logistic_regression.start_gradient_descent(train_X, train_Y, iterations=150)

# Use the trained model to make predictions on the validation set
predictions = logistic_regression.predict(validation_X, model_weights)

# Evaluate the predictions made by the model
logistic_regression.evaluate(validation_Y, predictions)

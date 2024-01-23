import numpy as np

def gradient_descent(train_X, train_Y, alpha = 0.001):
    thetas = np.zeros((3, 1))
    for i in range(500):
        preds = []
        for instance in train_X:
            preds.append(np.dot(thetas.T, instance) + 1)
        preds = np.array(preds)
        sigmoids = 1 / (1 + np.power(np.e, -preds))
        for j in range(len(thetas)):
            gradient = (sigmoids - train_Y) * train_X[:, j:j+1]
            gradient = (1 / train_X.shape[0]) * np.sum(gradient)
            thetas[j] -= alpha * gradient
import numpy as np

# Gradient descent optimization algorithm that is used to get the best theta parameters for
# the logistic regression model
def start_gradient_descent(train_X, train_Y, iterations, alpha = 0.01):
    theta = np.zeros((1, 3))
    for i in range(iterations):
        z_l = np.matmul(theta, train_X.T) + 1
        pred_l = np.array(calculate_sigmoid(np.squeeze(z_l)))
        loss = compute_loss(train_Y, pred_l)
        gradient = calculate_gradient(train_X, train_Y, pred_l)
        for j in range(len(theta)):
            gradient = (pred_l - train_Y) * train_X[:, j:j+1]
            gradient = (1 / train_X.shape[0]) * np.sum(gradient)
            theta[j] -= alpha * gradient

# Calculate the sigmoid of each of the z-values in the list
def calculate_sigmoid(z_l):
    sigmoid_l = []
    for z in z_l:
        if z >= 0:
            sigmoid_l.append(1 / (1 + np.exp(-z)))
        else:
            sigmoid_l.append(np.exp(z) / (1 + np.exp(z)))
    return sigmoid_l

# Calculate the loss between the true labels and the predictions
def compute_loss(y_l, pred_l):
    loss = -np.mean(y_l * np.log(pred_l) + (1 - y_l) * np.log(1 - pred_l))
    return loss

# Calculate the derivative of the loss function of the logistic regression model
def calculate_gradient(x_l, y_l, pred_l):
    gradient = np.mean(np.matmul(x_l.T, (pred_l - y_l)))
    return gradient
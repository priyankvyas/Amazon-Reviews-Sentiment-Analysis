import numpy as np

# Gradient descent optimization algorithm that is used to get the best theta and bias parameters for
# the logistic regression model
def start_gradient_descent(train_X, train_Y, iterations, alpha = 0.01):
    theta = np.zeros((2,))
    bias = 0
    for i in range(iterations):
        z_l = np.matmul(theta, train_X.T) + bias
        pred_l = np.array(calculate_sigmoid(z_l))
        loss = compute_loss(train_Y, pred_l)
        gradient = calculate_gradient(train_X, train_Y, pred_l)
        theta = update_theta(alpha, theta, gradient)

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
    gradient_l = np.matmul((pred_l - y_l).T, x_l)
    gradient_l = np.array([np.mean(gradient) for gradient in gradient_l])
    return gradient_l

def update_theta(alpha, theta, gradient):
    return theta - alpha * gradient
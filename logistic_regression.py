import numpy as np


# Gradient descent optimization algorithm that is used to get the best theta parameters for
# the logistic regression model
def start_gradient_descent(train_x, train_y, iterations, alpha=0.01):
    theta = np.zeros(train_x.shape[1])
    bias = 1
    for i in range(iterations):
        z_l = np.matmul(theta, train_x.T) + bias
        pred_l = np.array(calculate_sigmoid(z_l))
        loss = compute_loss(train_y, pred_l)
        gradient_weight = calculate_gradient_weight(train_x, train_y, pred_l)
        theta = update_theta(alpha, theta, gradient_weight)
        pred_l = [1.0 if pred > 0.5 else 0.0 for pred in pred_l]
        accuracy = np.mean(train_y == pred_l)
        print("Epoch {0}: Training accuracy = {1} Loss = {2}".format(i, accuracy, loss))
    return theta


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
    loss = -np.mean(y_l * np.log(pred_l + 1e-8) + (1 - y_l) * np.log(1 - pred_l + 1e-8))
    return loss


# Calculate the derivative of the loss function of the logistic regression model
def calculate_gradient_weight(x_l, y_l, pred_l):
    gradient_l = np.matmul((pred_l - y_l).T, x_l)
    gradient_l = np.array([np.mean(gradient) for gradient in gradient_l])
    return gradient_l


# Update the weights of the logistic regression model
def update_theta(alpha, theta, gradient):
    return theta - alpha * gradient


# Make predictions on a given instance or instances using the logistic regression model
def predict(test_l, theta):
    bias = 1
    z_l = np.matmul(theta, test_l.T) + bias
    pred_l = np.array(calculate_sigmoid(z_l))
    pred_l = np.array([1.0 if pred > 0.5 else 0.0 for pred in pred_l])
    return pred_l


# Evaluate the results given by the model against the true labels
def evaluate(true_labels, pred_labels):
    loss = compute_loss(true_labels, pred_labels)
    accuracy = np.mean(true_labels == pred_labels)
    print("Training accuracy = {0} Loss = {1}".format(accuracy, loss))
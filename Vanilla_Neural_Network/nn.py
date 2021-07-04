import csv
import numpy as np
import pandas as pd

#### data preprocessing ####
def import_train_file():
    train = pd.read_csv('train.csv')
    train = np.array(train)
    m, n = train.shape
    np.random.shuffle(train)

    train_train = train[1000:m].T
    y_train_train = train_train[0]
    X_train_train = train_train[1:n]
    X_train_train = X_train_train/255

    train_val = train[0:1000].T
    y_train_val = train_val[0]
    X_train_val = train_val[1:n]
    X_train_val = X_train_val/255
    return X_train_train, y_train_train, X_train_val, y_train_val

def import_test_file():
    test = pd.read_csv('test.csv')
    test = np.array(test)
    X_test = test/255
    return X_test.T

#### helper function ####
def sigmoid(Z):
    return 1/(1+np.exp(-Z))


def relu(Z):
    return np.maximum(0,Z)


def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)


def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

#### initiate parameters ####
def init_parameters(layer_dim):
    """
    Return dictionary of layers
    """
    np.random.seed(1)
    total_layer = len(layer_dim)
    parameter = {}
    for i in range(1, total_layer):
        parameter['W' + str(i)] = np.random.randn(layer_dim[i], layer_dim[i-1]) * 0.01
        parameter['b' + str(i)] = np.random.randn(layer_dim[i], 1)

    return parameter


#### forward ####
def forward_layer(A_prev, W, b):
    """
    Simple forward layer
    """
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)
    return Z, cache


def forward_activation_layer(A_prev, W, b, activation):
    """
    Forward layer with activation
    """
    Z, prev_cache = forward_layer(A_prev, W, b)
    if activation == 'relu':
        A = relu(Z)
    elif activation == 'sigmoid':
        A = sigmoid(Z)
    current_cache = Z
    cache = (prev_cache, current_cache)
    return A, cache


def forward_full(data, parameter):
    """
    Forward full step
    """
    caches = []
    A_curr = data
    L = len(parameter) // 2
    
    # Relu
    for i in range(1, L):
        A_prev = A_curr
        A_curr, cache = forward_activation_layer(A_prev, parameter['W' + str(i)], parameter['b' + str(i)], 'relu')
        caches.append(cache)

    A_final, cache = forward_activation_layer(A_curr,  parameter['W' + str(L)], parameter['b' + str(L)], 'sigmoid')
    caches.append(cache)
    return A_final, caches


### cost ###
def loss(A_final, Y):
    Y_one_hot_vec = one_hot(Y)
    loss_sample = (np.log(A_final) * Y_one_hot_vec).sum(axis=1)
    return -np.mean(loss_sample)


#### backward ####
def backward_layer(dZ, cache):
    """
    Simple backward
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def backward_activation_layer(dA, cache, activation):
    """
    Backward layer with activation
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    
    dA_prev, dW, db = backward_layer(dZ, linear_cache)
    
    return dA_prev, dW, db


def backward_full(AL, Y, caches):
    gradients = {}
    L = len(caches)
    m = AL.shape[1]
    Y = one_hot(Y).reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L-1]
    gradients["dA" + str(L-1)], gradients["dW" + str(L)], gradients["db" + str(L)] = backward_activation_layer(dAL, current_cache, "sigmoid")

    for i in reversed(range(L-1)):
        current_cache = caches[i]
        dA_prev_loop, dW_loop, db_loop = backward_activation_layer(gradients["dA" + str(i+1)], current_cache, "relu")
        gradients["dA" + str(i)] = dA_prev_loop
        gradients["dW" + str(i + 1)] = dW_loop
        gradients["db" + str(i + 1)] = db_loop
    
    return gradients


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - (learning_rate * grads["dW" + str(l+1)])
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - (learning_rate * grads["db" + str(l+1)])
    return parameters

def get_predictions(A2):
    return np.argmax(A2,0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, iterations, dimension, alpha):
    np.random.seed(1)

    parameter = init_parameters(dimension)
    for i in range(iterations):
        A_final, caches = forward_full(X, parameter)
        gradients = backward_full(A_final, Y, caches)
        parameter = update_parameters(parameter, gradients, alpha)
        cost = loss(A_final, Y)
        if i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
            print("\n accuracy: ",  get_accuracy(get_predictions(A_final), Y))

    return parameter


### write data ###
def write(y_test):
    header = ['ImageId', 'Label']
    with open('result.csv', 'w', newline='', encoding = 'UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for index, value in enumerate(y_test):
            writer.writerow([index+1, value])
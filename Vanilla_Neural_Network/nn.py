import numpy as np
import csv

#### data preprocessing ####
def import_file(csvfile):
    pass

def scale(data):
    pass

#### helper function ####
def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

#### initiate parameters ####
def init_parameters(layer_dim, seed):
    """
    Return dictionary of layers
    """
    np.random.seed(seed)
    total_layer = len(layer_dim)
    parameter = {}
    for i in range(1, total_layer):
        parameter['W' + str(i)] = np.random.randn(layer_dim[i], layer_dim[i-1]) * 0.1
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

    A_final, cache = forward_activation_layer(A_curr,  parameter['W' + str(i)], parameter['b' + str(i)], 'sigmoid')
    caches.append(cache)
    return A_final, caches
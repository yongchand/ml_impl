import numpy as np

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

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
    if activation == "relu":
        A = relu(Z)
    elif activation == "sigmoid":
        A = sigmoid(Z)
    current_cache = Z
    cache = (prev_cache, current_cache)
    return A, cache

# def forward_full(data, parameters):
#     cache = []

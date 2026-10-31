import numpy as np

def sigmoid(x):
    return (1/(1+np.exp(-x))) * (1-(1/(1+np.exp(-x))))

def relu(x):
    return np.where(x > 0, 1, 0)
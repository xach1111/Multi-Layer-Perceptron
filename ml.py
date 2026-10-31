import numpy as np
import activations
import activationsDerivatives
class mlp():
    def __init__(self, layers, lr = 0.1, maxIterations = 10000, hiddenLayerActivation = "relu", outputLayerActivation = "sigmoid"):
        self.layers = layers
        self.lr = lr
        self.maxIterations = maxIterations
        self.hiddenLayerActivation = getattr(activations, hiddenLayerActivation)
        self.outputLayerActivation = getattr(activations, outputLayerActivation)
        self.hiddenLayerActivationDerivative = getattr(activationsDerivatives, hiddenLayerActivation)
        self.outputLayerActivationDerivative = getattr(activationsDerivatives, outputLayerActivation)
        self.W = [np.random.randn(layers[i], layers[i+1]) * 0.1 for i in range(len(layers) -1)]
        self.B = [np.zeros((1, layers[i+1])) for i in range(len(layers) -1)]
    
    def fit(self, X_train, y_train):
        for iter in range(self.maxIterations):
            U, V = [], []
            # Forward progagate
            v = X_train
            for i, (w, b) in enumerate(zip(self.W, self.B)):
                u = np.dot(v, w) + b
                v = self.outputLayerActivation(u) if i == min(len(self.W),len(self.B))-1 else self.hiddenLayerActivation(u)
                U.append(u)
                V.append(v)

            # Back propagation
            E = v - y_train
            S = []
            for i, (w, u) in enumerate(zip(reversed(self.W), reversed(U))):
                s = E * (self.outputLayerActivationDerivative(u) if i == 0 else self.hiddenLayerActivationDerivative(u))
                S.insert(0, s)
                E = np.dot(s,w.T)
            # Calculate Difference in Weights and Biases
            DW = []
            DB = []
            for v, s in zip([X_train] + V[:-1], S):
                DW.append(np.dot(v.T ,s))
                DB.append(np.sum(s, axis=0, keepdims=True))
            
            #change Weights and Biases
            for i in range(len(self.W)):
                self.W[i] -= self.lr * DW[i]
                self.B[i] -= self.lr * DB[i]
    
    def predict(self, X_train):
        v = X_train
        for i, (w, b) in enumerate(zip(self.W, self.B)):
            u = np.dot(v, w) + b
            v = self.outputLayerActivation(u) if i == min(len(self.W),len(self.B))-1 else self.hiddenLayerActivation(u)
        return v
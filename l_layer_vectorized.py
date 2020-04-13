import tensorflow as tf
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
import numpy as np
import math
import matplotlib.pyplot as plt
import time

def shape_features(x_test,x_train, y_test, y_train):
    x_test = x_test/255
    x_train = x_train/255
    x_test = x_test.reshape(x_test.shape[0],-1).T
    x_train = x_train.reshape(x_train.shape[0],-1).T

    y_train = keras.utils.to_categorical(y_train,10)
    y_test = keras.utils.to_categorical(y_test,10)
    y_train = y_train.reshape(y_train.shape[0],-1).T
    y_test = y_test.reshape(y_test.shape[0],-1).T

    return x_test, x_train, y_test, y_train

def tanh(X):
    return (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))

def softmax(X):
    t = np.exp(X)
    t_sum = np.sum(t,axis=0,keepdims=True)
    return t/t_sum

def d_softmax(X):
    pass

def d_tanh(A):
    return 1 - A**2

def relu(X):
    X[X<0] = 0
    return X

def d_relu(A):
    A[A<0] = 0
    A[A!=0] = 1
    return A

def sigmoid(X):
    return 1/(1+np.exp(-X))

def d_sigmoid(A):
    return A * (1-A)

def cost_function(m,y, y_pred):
    y = y.astype(float)
    y_pred = y_pred.astype(float)
    return np.squeeze(-1 * 1/m * np.sum(np.sum(y*np.log(y_pred),axis=0,keepdims=True),axis=1,keepdims=True))
    #return float(np.squeeze(-1 * 1/m * np.sum(y * np.log(y_pred) + (1-y) * np.log(1-y_pred))))

def plot_cost(cost):
    plt.plot(cost)
    plt.ylabel("Cost")
    plt.xlabel("itterations")
    plt.show()


class vanilla_ml():
    def __init__(self, l_dims,activation_functions,optimiser = "adam"):

        """
        Initalize network parameters based on the givven l_dims list
        l_dims = python list containing th number of nodes in each leayer
        len(l_dims) = Number of layer in network

        Returns:
        self.parameters = python dictonary containing the weights an biases
        self.grads = python dictonary containing the derivate of weights and biases

        """
        self.parameters = {}
        self.grads = {}
        self.n_layers = len(l_dims)-1
        self.activation_functions = activation_functions
        self.optimiser = optimiser

        for l in range(self.n_layers):
            self.parameters["W" + str(l+1)] = np.random.randn(l_dims[l+1],l_dims[l]) * np.sqrt(1/(l_dims[l]-1))
            self.parameters["B" + str(l+1)] = np.zeros((l_dims[l+1],1))
            self.grads["dW" + str(l+1)] = np.zeros((l_dims[l+1],l_dims[l]))
            self.grads["dB" + str(l+1)] = np.zeros((l_dims[l+1],1))

        if self.optimiser == "adam":
            self.V = {}
            self.S = {}
            for l in range(self.n_layers):
                self.V["dW" + str(l+1)] = np.zeros(self.parameters["W" + str(l+1)].shape)
                self.V["dB" + str(l+1)] = np.zeros(self.parameters["B" + str(l+1)].shape)
                self.S["dW" + str(l+1)] = np.zeros(self.parameters["W" + str(l+1)].shape)
                self.S["dB" + str(l+1)] = np.zeros(self.parameters["B" + str(l+1)].shape)

    def forward_propogate(self,X):

        """
        Forward propogation based on the given activation_functions dictonary
        self.activation_functions = python dictonary containing the used activation function for each layer

        Reutrns:
        self.Z = python dictonary containing the Zl for each layer
        self.A = python dictonary containing the activations of each layer

        """
        self.Z = {}
        self.A = {}
        self.A["A0"] = X
        for l in range(self.n_layers):
            self.Z["Z" + str(l+1)] = self.parameters["W" + str(l+1)] @ self.A["A" + str(l)] + self.parameters["B" + str(l+1)]
            if self.activation_functions[str(l+1)] == "relu":
                self.A["A" + str(l+1)] = relu(self.Z["Z" + str(l+1)])
            elif  self.activation_functions[str(l+1)] == "sigmoid":
                self.A["A" + str(l+1)] = sigmoid(self.Z["Z" + str(l+1)])
            elif  self.activation_functions[str(l+1)] == "softmax":
                self.A["A" + str(l+1)] = softmax(self.Z["Z" + str(l+1)])


        

    def backward_propagate(self, Y, X, m):

        last_layer = str(self.n_layers)
        self.dZ = {}
        self.dZ["dZ" + last_layer] = self.A["A" + last_layer] - Y
        self.grads["dW" + last_layer] = 1/m * self.dZ["dZ" + last_layer] @ self.A["A" + str(int(last_layer)-1)].T
        self.grads["dB" + last_layer] = 1/m * np.sum(self.dZ["dZ" + last_layer], axis=1, keepdims=True)

        for l in range(int(last_layer)-1,0,-1):            
            self.dZ["dZ" + str(l)] = self.parameters["W" + str(l+1)].T @ self.dZ["dZ" + str(l+1)]
            if self.activation_functions[str(l)] == "relu":
                self.dZ["dZ" + str(l)] = self.dZ["dZ" + str(l)] * d_relu(self.A["A" + str(l)])
            elif  self.activation_functions[str(l)] == "sigmoid":
                self.dZ["dZ" + str(l)] = self.dZ["dZ" + str(l)] * d_sigmoid(self.A["A" + str(l)])
            elif  self.activation_functions[str(l)] == "softmax":
                self.dZ["dZ" + str(l)] = self.dZ["dZ" + str(l)] * d_softmax(self.A["A" + str(l)])
            self.grads["dW" + str(l)] = 1/m * self.dZ["dZ" + str(l)] @ self.A["A" + str(l-1)].T
            self.grads["dB" + str(l)] = 1/m * np.sum(self.dZ["dZ" + str(l)], axis=1, keepdims=True)


    def update_parameters_grad(self,learn_rate):
        for l in range(self.n_layers):
            self.parameters["dW" + str(l+1)] -= learn_rate * self.grads["dW" + str(l+1)]
            self.parameters["dB" + str(l+1)] -= learn_rate * self.grads["dB" + str(l+1)]


    def update_parameters_adam(self,learn_rate,beta1,beta2,epsilon,t):

        v_corrected = {}
        s_corrected = {}
        for l in range(self.n_layers):
            self.V["dW" + str(l+1)] = self.V["dW" + str(l+1)] * beta1 + self.grads["dW" + str(l+1)] * (1-beta1)
            self.V["dB" + str(l+1)] = self.V["dB" + str(l+1)] * beta1 + self.grads["dB" + str(l+1)] * (1-beta1)

            # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
            v_corrected["dW" + str(l+1)] = self.V["dW" + str(l+1)]/(1-beta1**t)
            v_corrected["dB" + str(l+1)] = self.V["dB" + str(l+1)]/(1-beta1**t)

            # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
            self.S["dW" + str(l+1)] = self.S["dW" + str(l+1)] * beta2 + self.grads["dW" + str(l+1)]**2 * (1-beta2)
            self.S["dB" + str(l+1)] = self.S["dB" + str(l+1)] * beta2 + self.grads["dB" + str(l+1)]**2 * (1-beta2)

            # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
            s_corrected["dW" + str(l+1)] = self.S["dW" + str(l+1)]/(1-beta2**t)
            s_corrected["dB" + str(l+1)] = self.S["dB" + str(l+1)]/(1-beta2**t)

            # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
            self.parameters["W" + str(l+1)] -= learning_rate * (v_corrected["dW" + str(l+1)]/ (np.sqrt(s_corrected["dW" + str(l+1)])+epsilon))   
            self.parameters["B" + str(l+1)] -= learning_rate * (v_corrected["dB" + str(l+1)]/ (np.sqrt(s_corrected["dB" + str(l+1)])+epsilon))

    def gradient_descent(self, X, Y, learn_rate, m,**kwargs):
        
        #Do forward and bacward propogation
        self.forward_propogate(X)
        self.backward_propagate(Y, X, m)

        #Adjust weights and biases
        if self.optimiser == "adam":
            beta1 = kwargs.get('beta1')
            beta2 = kwargs.get('beta2')
            epsilon = kwargs.get('epsilon')
            t = kwargs.get('t')
            self.update_parameters_adam(learn_rate,beta1,beta2,epsilon,t)
        elif self.optimiser == "grad":
            self.update_parameters_grad(learn_rate)

    def random_mini_batches(self,X, Y, mini_batch_size = 264, seed = 0):
        """
        Creates a list of random minibatches from (X, Y)
        
        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
        mini_batch_size -- size of the mini-batches, integer
        
        Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
        """                   
        m = X.shape[1]                  
        mini_batches = []
            
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((10,m))
        
        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(m/mini_batch_size)
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:,(k*mini_batch_size):(mini_batch_size*(k+1))]
            mini_batch_Y = shuffled_Y[:,(k*mini_batch_size):(mini_batch_size*(k+1))]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:

            mini_batch_X = shuffled_X[:,-(m-num_complete_minibatches*mini_batch_size):]
            mini_batch_Y = shuffled_Y[:,-(m-num_complete_minibatches*mini_batch_size):]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
            
        return mini_batches   
        
    def train_network(self, X_test, Y_test, learn_rate, mini_batch_size,n_epoch,**kwargs):
        
        cost = []        
        for itter in range(1,n_epoch):
            
            mini_batches = self.random_mini_batches(X=X_test,Y=Y_test,mini_batch_size=mini_batch_size)
            t = 0

            for mini_batch in mini_batches:
                (X,Y) = mini_batch
                t += 1
                if self.optimiser == "adam":
                    beta1 = kwargs.get('beta1')
                    beta2 = kwargs.get('beta2')
                    epsilon = kwargs.get('epsilon')
                    m = X.shape[1]
                    self.gradient_descent(X, Y, learn_rate, m,beta1=beta1,beta2=beta2,epsilon=epsilon,t=t)
                elif self.optimiser == "grad":
                    m = X.shape[1]
                    self.gradient_descent(X, Y, learn_rate, m)
                cost.append(cost_function(m, Y, self.A["A" + str(self.n_layers)]))

            if itter % 1 == 0:
                print("{}/{} Epochs finished\n Cost: {}\n-----------------".format(itter,n_epoch, cost[-1]))

        return cost

    def predict(self, X, Y):

        m_examples = X.shape[1]
        self.forward_propogate(X)
        Y_that = (self.A["A" + str(self.n_layers)]==self.A["A" + str(self.n_layers)].max(axis=0, keepdims=1)).astype(float)

        prediction = {"Y_that":Y_that,
                      "Accuracy":0,
                      "Cost":0.00001}
        prediction["Cost"] = cost_function(m_examples,Y,Y_that)
        Y_comp = np.abs(Y_that - Y).sum(axis=0, keepdims=True)
        prediction["Accuracy"] = np.count_nonzero(Y_comp==0)/m_examples
        
        return prediction

#Setting Hyperparameters
layer_dimensions = [784,350,300,10]
optimiser = "adam"
activation_functions = {"1":"relu","2":"relu","3":"softmax"}
learning_rate = 0.008
mini_batch_size = 514
n_epoch = 35
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

#Load and shape features and labels
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X_test, X_train, Y_test, Y_train = shape_features(x_test,x_train, y_test, y_train)

#Train network
start = time.time()
Deep_Network = vanilla_ml(layer_dimensions,activation_functions,optimiser)
cost = Deep_Network.train_network(X_train, Y_train, learning_rate, mini_batch_size, n_epoch, beta1=beta1,beta2=beta2,epsilon=epsilon)
plot_cost(cost)
prediction = Deep_Network.predict(X_test,Y_test)

print("Run Time: {} Minutes".format((time.time()-start)/60))
print(prediction)


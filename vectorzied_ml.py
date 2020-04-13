import tensorflow as tf
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import math
import matplotlib.pyplot as plt

def shape_features(x_test,x_train, y_test, y_train):
    x_test = x_test/255
    x_train = x_train/255
    x_test = x_test.reshape(x_test.shape[0],-1).T
    x_train = x_train.reshape(x_train.shape[0],-1).T

    y_train = keras.utils.to_categorical(y_train,nodes_3)
    y_test = keras.utils.to_categorical(y_test,nodes_3)
    y_train = y_train.reshape(y_train.shape[0],-1).T
    y_test = y_test.reshape(y_test.shape[0],-1).T

    return x_test, x_train, y_test, y_train

def tanh(X):
    return (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))

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
    return -1 * 1/m * np.sum(y * np.log(y_pred) + (1-y) * np.log(1-y_pred))

def plot_cost(cost):
    plt.plot(cost)
    plt.ylabel("Cost")
    plt.xlabel("itterations")
    plt.show()


class vanilla_ml():
    def __init__(self):
        self.W1 = np.random.randn(nodes_1,nodes_0) * init_const
        self.W2 = np.random.randn(nodes_2,nodes_1) * init_const
        self.W3 = np.random.randn(nodes_3,nodes_2) * init_const
        self.B1 = np.zeros((nodes_1,1))
        self.B2 = np.zeros((nodes_2,1))
        self.B3 = np.zeros((nodes_3,1))

    def forward_propogate(self,X):
        self.Z1 = self.W1 @ X + self.B1
        self.A1 = relu(self.Z1)
        self.Z2 = self.W2 @ self.A1 + self.B2
        self.A2 = relu(self.Z2)
        self.Z3 = self.W3 @ self.A2 + self.B3
        self.A3 = sigmoid(self.Z3)

        

    def backward_propagate(self, Y, X, m):
        self.dZ3 = self.A3 - Y
        self.dW3 = 1/m * self.dZ3 @ self.A2.T
        self.dB3 = 1/m * np.sum(self.dZ3, axis=1, keepdims=True)

        self.dZ2 = self.W3.T @ self.dZ3 * d_relu(self.A2)
        self.dW2 = 1/m * self.dZ2 @ self.A1.T
        self.dB2 = 1/m * np.sum(self.dZ2, axis=1, keepdims=True)

        self.dZ1 = self.W2.T @ self.dZ2 * d_relu(self.A1)
        self.dW1 = 1/m * self.dZ1 @ X.T
        self.dB1 = 1/m * np.sum(self.dZ1, axis=1, keepdims=True)

    def gradient_descent(self, X, Y, learn_rate, m):
        
        #Do forward and bacward propogation
        self.forward_propogate(X)
        self.backward_propagate(Y, X, m)

        #Adjust weights and biases
        self.W1 -= learn_rate * self.dW1
        self.W2 -= learn_rate * self.dW2
        self.W3 -= learn_rate * self.dW3

        self.B1 -= learn_rate * self.dB1
        self.B2 -= learn_rate * self.dB2
        self.B3 -= learn_rate * self.dB3
        

    def train_network(self, X_test, Y_test, learn_rate, m, itterations):
        
        cost = []
        n_branches = int(X_test.shape[1]/m) + 1
        
        for itter in range(1,itterations):
            
            m = X_test.shape[1]
            self.gradient_descent(X_test, Y_test, learn_rate, m)
            cost.append(cost_function(m, Y_test, self.A3 ))

            # for i in range(1,n_branches):
                
            #     X = X_test[:,(i-1)*m:i*m]
            #     Y = Y_test[:,(i-1)*m:i*m]
            #     self.gradient_descent(X, Y, learn_rate, m)
            #     cost.append(cost_function(m, Y, self.A3 ))
            
            if itter % 10 == 0:
                print("{}/{} Itterations finished\n Cost: {}\n-----------------".format(itter,itterations, cost[-1]))

        return cost

    def predict(self, X, Y):

        m_examples = X.shape[1]
        self.forward_propogate(X)
        Y_that = (self.A3==self.A3.max(axis=0, keepdims=1)).astype(float)

        prediction = {"Y_that":Y_that,
                      "Accuracy":0,
                      "Cost":0}
        prediction["Cost"] = np.squeeze(cost_function(m_examples,Y,Y_that))
        Y_comp = np.abs(Y_that - Y).sum(axis=0, keepdims=True)
        prediction["Accuracy"] = np.count_nonzero(Y_comp==0)/m_examples
        
        return prediction

#Setting number of nodes
nodes_0 = 784
nodes_1 = 150
nodes_2 = 150
nodes_3 = 10
init_const = 0.01
learning_rate = 0.1
m = 1000
itterations = 3000

#Load and shape features and labels
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X_test, X_train, Y_test, Y_train = shape_features(x_test,x_train, y_test, y_train)

#Train network
Deep_Network = vanilla_ml()
cost = Deep_Network.train_network(X_train, Y_train, learning_rate, m, itterations)
plot_cost(cost)
prediction = Deep_Network.predict(X_test,Y_test)

print("stop")


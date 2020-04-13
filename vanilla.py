import tensorflow as tf
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import math
import matplotlib.pyplot as plt


def sigmoid(x):
    try:
        return 1/(1+math.exp(-x))
    except OverflowError:
        x = float('inf')
        return 1/(1+math.exp(-x))
    ##need check

def sigmoid_deriv(x):
    return sigmoid(x) * (1-sigmoid(x))

def leaky_relu(x):
    if x < 0:
        return x*0.1
    else:
        return x

def leaky_relu_deriv(x):

    for  i in range(x.size):
        if x[i] < 0:
            y = 0.1
        else:
            x[i] = 1
    return x

class vanilla_ml():
    def __init__(self):
        self.w1 = np.random.rand(layer_1,layer_0)*np.sqrt(1/layer_0)
        self.w2 = np.random.rand(layer_2,layer_1)*np.sqrt(1/layer_1)
        self.w3 = np.random.rand(layer_3,layer_2)*np.sqrt(1/layer_2)
        self.b1 = np.zeros(layer_1)
        self.b2 = np.zeros(layer_2)
        self.b3 = np.zeros(layer_3)
        self.o1 = np.empty(layer_1,float)
        self.o2 = np.empty(layer_2,float)
        self.o3 = np.empty(layer_3,float)
        self.w1_adj = np.zeros((layer_1,layer_0))
        self.w2_adj = np.zeros((layer_2,layer_1))
        self.w3_adj = np.zeros((layer_3,layer_2))
        self.b1_adj = np.zeros(layer_1)
        self.b2_adj = np.zeros(layer_2)
        self.b3_adj = np.zeros(layer_3)

    def calc_output(self,input):
        
        self.i1 =np.dot(self.w1,input)+self.b1
        if funct_1 == "relu":
            for i in range(layer_1):
                self.o1[i] = leaky_relu(self.i1[i])
        elif funct_1 == "sigmoid":
            for i in range(layer_1):
                self.o1[i] = sigmoid(self.i1[i])
        
        self.i2 = np.dot(self.w2,self.o1)+self.b2
        if funct_2 == "relu":
            for i in range(layer_2):
                self.o2[i] = leaky_relu(self.i2[i])
        elif funct_2 == "sigmoid":
            for i in range(layer_2):
                self.o2[i] = sigmoid(self.i2[i])
        
        self.i3 = np.dot(self.w3,self.o2)+self.b3
        if funct_3 == "relu":
            for i in range(layer_3):
                self.o3[i] = leaky_relu(self.i3[i])
        elif funct_3 == "sigmoid":
            for i in range(layer_3):
                self.o3[i] = sigmoid(self.i3[i])

        return self.o3
    
    def calc_loss(self, out, target):
        return  np.sum((target-out)**2)
        
    def back_prop(self,x_t, y_t):
        
        out = self.calc_output(x_t)
        
        #Calculation the gradient of W3
        self.E_O3 =  2 * (y_t - out)

        if funct_3 == "relu":
            self.O3_I3 = leaky_relu_deriv(self.o3)
        elif funct_3 == "sigmoid":
            self.O3_I3 = self.o3 * (1-self.o3)

        self.I3_w3 = self.o2
        for i in range(layer_3):
            self.w3_adj[i] += self.E_O3[i] * self.O3_I3[i] * self.I3_w3

        #Calculation the gradient of B3
        self.b3_adj += self.E_O3 * self.O3_I3


        #Calcultion the gradient of W2
        self.E_I3 = self.E_O3 * self.O3_I3
        self.I3_O2 = np.empty((layer_2,layer_3),float)
        for i in range(layer_2):
            self.I3_O2[i] = self.w3[:,i]

        self.E_O2 = np.empty(layer_2,float)
        for i in range (layer_2):
            self.E_O2[i] = np.sum(self.E_I3*self.I3_O2[i])
            #check
        
        if funct_2 == "relu":
            self.O2_I2 = leaky_relu_deriv(self.o2)
        elif funct_2 == "sigmoid":
            self.O2_I2 = self.o2*(1-self.o2)

        self.I2_w2 = self.o1
        for i in range(layer_2):
            self.w2_adj[i] += self.E_O2[i] * self.O2_I2[i] * self.I2_w2


        #Calculation the gradient of B2
        self.b2_adj += self.E_O2 * self.O2_I2


        #Calculatiom of the gradient of w1
        self.E_I2 = self.E_O2 * self. O2_I2
        self.I2_O1 = np.empty((layer_1,layer_2),float)
        for i in range(layer_1):
            self.I2_O1[i] = self.w2[:,i]

        self.E_O1 = np.empty(layer_1,float)
        for i in range(layer_1):
            self.E_O1[i] = np.sum(self.E_I2*self.I2_O1[i])

        if funct_1 == "relu":
            self.O1_I1 = leaky_relu_deriv(self.o1)
        elif funct_1 == "sigmoid":
            self.O1_I1 = self.o1 * (1-self.o1)  

        self.I1_w1 = x_t
        for i in range(layer_1):
            self.w1_adj[i] += self.E_O1[i] * self.O1_I1[i] * self.I1_w1

        #Calculation the gradient of B1
        self.b1_adj += self.E_O1 * self.O1_I1



    def train_network(self,X_test, Y_test, learn_rate):
        
        n_branches = int(len(X_test)/branch_size)
        for i in range(n_branches):
            x_branch = X_test[branch_size*i:min((i+1)*branch_size,len(X_test))]
            y_branch = Y_test[branch_size*i:min((i+1)*branch_size,len(X_test))]

            #Error before adjust
            err_bef = 0
            for branch in range(len(x_branch)):
                out = self.calc_output(np.concatenate(x_branch[branch],axis=None))
                err_bef += self.calc_loss(out, y_branch[branch])
            err_bef = err_bef / branch_size 


            #Train network on each example in branch
            for branch in range(len(x_branch)):
                self.back_prop(np.concatenate(x_branch[branch],axis=None),y_branch[branch])
            
            #Adjusting weights and biases
            self.w1 -= learn_rate * self.w1_adj / branch_size
            self.w2 -= learn_rate * self.w2_adj / branch_size
            self.w3 -= learn_rate * self.w3_adj / branch_size
            self.w1_adj = np.zeros((layer_1,layer_0))
            self.w2_adj = np.zeros((layer_2,layer_1))
            self.w3_adj = np.zeros((layer_3,layer_2))

            self.b1 -= learn_rate * self.b1_adj / branch_size
            self.b2 -= learn_rate * self.b2_adj / branch_size
            self.b3 -= learn_rate * self.b3_adj / branch_size
            self.b1_adj = np.zeros(layer_1)
            self.b2_adj = np.zeros(layer_2)
            self.b3_adj = np.zeros(layer_3)
            
            #Error after adjust
            err = 0
            for branch in range(len(x_branch)):
                out = self.calc_output(np.concatenate(x_branch[branch],axis=None))
                err += self.calc_loss(out, y_branch[branch])
            err = err / branch_size 

            print("{}/{} train compleated.".format(i,n_branches), " Error before: {}".format(err_bef)," Error  after: {}".format(err))

        return self

        


mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test=255
x_train = x_train/255
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

layer_0 = 784
layer_1 = 160
layer_2 = 160
layer_3 = 10

branch_size = 100
l_rate = 0.1
n_input = 60000

funct_1 = "sigmoid"
funct_2 = "sigmoid"
funct_3 = "sigmoid"

network = vanilla_ml()
network = network.train_network(x_train[:n_input],y_train[:n_input],l_rate)



print("stop")

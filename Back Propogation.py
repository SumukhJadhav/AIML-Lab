import numpy as np

X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)     # X = (hours sleeping, hours studying)
y = np.array(([92], [86], [89]), dtype=float)           # y = score on test


# scale units
X = X/np.amax(X, axis=0)        # maximum of X array
y = y/100                       # max test score is 100

#parameters
inputSize = 2
outputSize = 1
hiddenSize = 3

#weights
W1 = np.random.randn(inputSize, hiddenSize)        # (3x2) weight matrix from input to hidden layer
W2 = np.random.randn(hiddenSize, outputSize)       # (3x1) weight matrix from hidden to output layer

                          # Parameters
def forward(X):
        global z2                     #forward propagation through our network
        z = np.dot(X,W1)               # dot product of X (input) and first set of 3x2 weights
        z2 = sigmoid(z)            # activation function
        z3 = np.dot(z2, W2)        # dot product of hidden layer (z2) and second set of 3x1 weights
        o = sigmoid(z3)
                         # final activation function
        return o 

def sigmoid(s):
    return 1/(1+np.exp(-s))     # activation function 

def sigmoidPrime(s):
    return s * (1 - s)          # derivative of sigmoid

def backward(X, y, o):
    
    global W1, W2       
    o_error = y - o        # error in output
    o_delta = o_error*sigmoidPrime(o) # applying derivative of sigmoid to 
    z2_error = o_delta.dot(W2.T)    # z2 error: how much our hidden layer weights contributed to output error
    z2_delta = z2_error*sigmoidPrime(z2) # applying derivative of sigmoid to z2 error
    W1 += X.T.dot(z2_delta)       # adjusting first set (input --> hidden) weights
    W2 += z2.T.dot(o_delta)  # adjusting second set (hidden --> output) weights

def train (X, y):
    o = forward(X)
    backward(X, y, o)


print ("\nInput: \n" + str(X))
print ("\nActual Output: \n" + str(y)) 
print ("\nPredicted Output: \n" + str(forward(X)))
print ("\nLoss: \n" + str(np.mean(np.square(y - forward(X)))))     # mean sum squared loss)
train(X, y)


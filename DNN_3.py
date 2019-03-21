import tensorflow as tf #Ok
import matplotlib #Ok
matplotlib.use('Agg') #??
import math #Ok
import numpy as np # import and define library as object
import pandas as pd #Ok
import h5py #Ok
import matplotlib.pyplot as plt #Ok

from tensorflow.python.framework import ops # import the functions from different locations
from tf_utils import random_mini_batches,normalize,predict # import the functions from different locations
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc # import the functions from different locations
from sklearn.utils import shuffle # import the functions from different locations
import pickle #import library


#Loading train data
train_path = 'Training_set3.csv' # change csv file for input
train_df = pd.read_csv(train_path,delimiter=',',dtype='float',header=None) #???
train_df = shuffle(train_df) # To randomise the training dataset
print (train_df.iloc[0:3,:].values) # for checking
Y_train = train_df.iloc[:,3].values # output layer (last column of the file)
X_train = train_df.iloc[:,0:3].values # input layer (first seven columns of file) -Exclude 7
#X_train = normalize(X_train)# use normalize function to normalize training data

print ('Training data shape')
print (X_train.shape)

X_valid = X_train[0:244,:] # percentage of data for validation (rows)
X_train = X_train[920:,:] # percentage of data to train (rows)
Y_valid = Y_train[0:244] # rows of last column
Y_train = Y_train[920:] # remaining rows of last column

#print ("saving the validation set")
#v = train_df.iloc[0:345,:]
#v.to_csv('RDF_40_valid.csv',sep=',', index=False)


# Number of features: Clarify with cuc again for lines 41 to 45

N = X_train.shape[1] # N is the number of features 
X_train = X_train.T # matrix transpose for features
X_valid = X_valid.T # matrix transpose  for features
Y_train = np.reshape(Y_train, (1,Y_train.shape[0])) # matrix transpose for output
Y_valid = np.reshape(Y_valid, (1,Y_valid.shape[0])) # matrix transpose for output

print ("number of training examples = " + str(X_train.shape[1]))
print ("number of validation examples = " + str(X_valid.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_validation shape: " + str(X_valid.shape))
print ("Y_validation shape: " + str(Y_valid.shape))


# In[ ]:


def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an feature vector : N
    n_y -- scalar, number of classes (attack - non attack: 2)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    ### START CODE HERE ### (approx. 2 lines)
    X = tf.placeholder(tf.float32, shape=[n_x, None], name='X')
    Y = tf.placeholder(tf.float32, shape=[n_y, None], name='Y')
    ### END CODE HERE ###
    
    return X, Y


# In[ ]:


X, Y = create_placeholders(N,1)
print ("X = " + str(X))
print ("Y = " + str(Y))


# In[ ]:


def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [2, N]
                        b1 : [2, 1]
                        W2 : [1, 2]
                        b2 : [1, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2
    """
    
    #tf.set_random_seed(1)                   # so that your "random" numbers match ours
        
    ### START CODE HERE ### (approx. 6 lines of code)
    W1 = tf.get_variable("W1", [N, N], initializer=tf.contrib.layers.xavier_initializer()) # input layer
    b1 = tf.get_variable("b1", [N, 1], initializer=tf.zeros_initializer()) # biased function to be more exact to the actual data (reduce errors)
    W2 = tf.get_variable("W2", [N, N], initializer=tf.contrib.layers.xavier_initializer()) # hidden layer
    b2 = tf.get_variable("b2", [N, 1], initializer=tf.zeros_initializer()) # biased function
    W3 = tf.get_variable("W3", [2, N], initializer=tf.contrib.layers.xavier_initializer()) # hidden layer
    b3 = tf.get_variable("b3", [2, 1], initializer=tf.zeros_initializer()) # biased function
    W4 = tf.get_variable("W4", [1, 2], initializer=tf.contrib.layers.xavier_initializer()) # output layer
    b4 = tf.get_variable("b4", [1, 1], initializer=tf.zeros_initializer()) # biased function

    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4} # dictionary. It has the name as "" on the LHS, and value on the RHS. 
    
    return parameters

def copy_parameters(parameters):
    """

    """
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    W4 = tf.convert_to_tensor(parameters["W4"])
    b4 = tf.convert_to_tensor(parameters["b4"])

    ### END CODE HERE ###

    new_parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4} # dictionary. It has the name as "" on the LHS, and value on the RHS. 
    
    return new_parameters

# In[ ]:


tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))


# In[ ]:


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z4 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']

    
    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                                # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                               # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                               # Z3 = np.dot(W3, a2) + b3
    A3 = tf.nn.relu(Z3)                                              # A3 = relu(Z3)
    Z4 = tf.add(tf.matmul(W4, A3), b4)                               # Z4 = np.dot(W4, a3) + b4
    A4 = tf.nn.relu(Z4)                                              # A4 = relu(Z4)

    ### END CODE HERE ###
    
    # Lines 171 to 178 is to reduce the complex input layer to a singular output
    return A4


# In[ ]:

#Clarify with Cuc further on the below section

tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(N, 1)
    parameters = initialize_parameters()
    A3 = forward_propagation(X, parameters)
    print("A3 = " + str(A3))


# In[ ]:


def compute_cost(A3, Y):
    
    predictions = tf.transpose(A3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.sqrt(tf.losses.absolute_difference(labels=Y, predictions=A3))) 

    return cost




tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(N, 1)
    parameters = initialize_parameters()
    A3 = forward_propagation(X, parameters)
    cost = compute_cost(A3, Y)
    print("cost = " + str(cost))


# In[ ]:

tf.reset_default_graph()

def model(X_train, Y_train, X_valid, Y_valid, learning_rate = 0.0001,
          num_epochs = 100, minibatch_size = 8, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = N, number of training examples = ?)
    Y_train -- test set, of shape (output size = 2, number of training examples = ?)
    X_test -- training set, of shape (input size = N, number of training examples = ?)
    Y_test -- test set, of shape (output size = 2, number of test examples = ?)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    #tf.set_random_seed(1)                             # to keep consistent results
    #seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters()
    
    #with open("Multiscale_modeling_0-Copy1.txt", "rb") as myFile:
    #    old_parameters = pickle.load(myFile)
    #    print ('loading old parameters successfully')
    
    #parameters = copy_parameters(old_parameters)
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    A3 = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(A3, Y)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = (int)(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            #seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , minibatch_cost = sess.run([optimizer, cost], 
                                             feed_dict={X: minibatch_X, 
                                                        Y: minibatch_Y})
                ### END CODE HERE ###
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 10 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.figure()
        plt.plot(np.log(np.squeeze(costs)))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.savefig('DNN_Multiscale_Modeling')

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        
        
        return parameters

    
parameters = model(X_train, Y_train, X_valid, Y_valid,num_epochs=1000,learning_rate=0.001,minibatch_size=8)
with open("Multiscale_modeling_"+str(0)+".txt", "wb") as myFile:
    pickle.dump(parameters, myFile)
    
valid_predict = predict(X_valid, parameters)

tf.reset_default_graph()
with tf.Session() as sess:
    V_Pre = tf.placeholder(tf.float32, shape=[Y_valid.shape[1], None], name='V_Pre')
    V_True = tf.placeholder(tf.float32, shape=[Y_valid.shape[1], None], name='V_True')
    V_Loss = compute_cost (V_Pre,V_True)
    valid_loss = sess.run(V_Loss, feed_dict={V_Pre:valid_predict.T, V_True:Y_valid.T})
    print ("valid_loss is: ")
    print (valid_loss)

np.savetxt('valid_predict.csv', valid_predict.T, delimiter=',')

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Hydraulic gradient_measured', fontsize = 15)
ax.set_ylabel('Hydraulic gradient_predicted', fontsize = 15)
ax.set_title('DNN_Multiscale_Modeling', fontsize = 20)
ax.scatter(Y_valid, valid_predict,c = 'b',s = 50)
ax.grid()
plt.savefig('DNN_Multiscale_Modeling_Prediction')


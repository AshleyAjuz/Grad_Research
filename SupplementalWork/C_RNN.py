import numpy as np
from tensorflow import keras
from SupplementalWork.evalGRURNN import computeMetrics

'''Function performs Hybrid CNN/RNN Model
Parameters include:
    -Training_data:
    -Training labels: true label for each training vector represented 
                      by one-hot vector y(x) = 0 for honest and y(x) = 1 for malicious
    -Testing_data:
    -Testing_labels:
    -N : Number of neurons/hidden units in each layer
    -B : Batch size for training the data points
    -I : Number of Epochs
'''
def evalC_RNN(X_train, y_train, X_test, y_test, N, B, I):

    X = np.array(X_train)
    y = np.array(y_train)
    X_t = np.array(X_test)

    #Instantiate the model
    c_rnn_mdl = keras.Sequential()

    #Build structure with layers
    c_rnn_mdl.add(keras.layers.Conv1D(32, 4, activation="relu", input_shape=(X.shape[1],1)))
    c_rnn_mdl.add(keras.layers.MaxPooling1D(2))
    c_rnn_mdl.add(keras.layers.GRU(N,return_sequences=True,reset_after = True))
    c_rnn_mdl.add(keras.layers.GRU(N,return_sequences=True,reset_after = True))
    c_rnn_mdl.add(keras.layers.GRU(N,return_sequences=True,reset_after = True))
    c_rnn_mdl.add(keras.layers.GRU(N,reset_after = True))
    c_rnn_mdl.add(keras.layers.Dense(2, activation="softmax"))

    #Define the loss and optimizer 
    loss = keras.losses.SparseCategoricalCrossentropy()
    optim = keras.optimizers.RMSprop(learning_rate= 0.001)
    metrics = ["accuracy"]

    #Model compliation
    c_rnn_mdl.compile(optim, loss, metrics)

    #Fit the model
    c_rnn_mdl.fit(X, 
                y, 
                shuffle = False,
                batch_size = B,
                epochs = I,
                verbose = 1,
                )
    
    #Predict output
    y_pred = np.round(c_rnn_mdl.predict(X_t, batch_size = B))
                       
    return(computeMetrics(y_test,y_pred))

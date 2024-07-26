# Importing the libraries
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



from sklearn.metrics import mean_squared_error, r2_score


from tensorflow import keras

def train_test_plot(dataset, tstart, tend, my_path, filename):
   
    fig, ax = plt.subplots() 

    data = dataset.T
    x = np.linspace(tstart,1, 408) 
    train_cust_0 = data[0][tstart:tend+1]
    test_cust_0 = data[0][tend:]

    ax.plot(x[tstart:tend+1],train_cust_0, color='blue', label="Traing") 
    ax.plot(x[tend:],test_cust_0, color='orange', label="Testing") 

    ax.set_xlabel('Time') 
    ax.set_ylabel('Energy Consumption') 
    ax.set_title('Training vs Testing Split')
    ax.legend()

    fig.savefig(my_path + filename)

def train_test_split(dataset, tstart, tend):
    train = dataset[tstart:tend]
    test = dataset[tend+1:]
    return np.array(train), np.array(test)


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def plot_predictions(test, predicted, my_path, filename):

    fig = plt.figure()
    plt.plot(test, color="gray", label="Real")
    plt.plot(predicted, color="red", label="Predicted")
    plt.title("Predicition")
    plt.xlabel("Time")
    plt.ylabel("Energy Consumption")
    plt.legend()


    fig.savefig(my_path + filename)

def plot_trainingLoss(model, my_path, filename):

    fig = plt.figure()
    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss'])
    plt.title('Training Loss vs Validation Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    fig.savefig(my_path + filename)


def evaluateTraining(test, predicted):
    #Redefine the test and predicted sets
    test = test.tolist()
    predicted = predicted.tolist()


    #Define variables
    Acc = 0

    #Calculate the root mean squared erro
    #This will be the threshold value to compare the difference between the predicted and real results to
    rmse = np.sqrt(mean_squared_error(test[:len(predicted)], predicted))
    r2 = r2_score(test[:len(predicted)], predicted)

    #Compute epsilon value that will act as standard deviation to add the rmse
    eps = rmse *.80


    for j in range(len(predicted)):
        diff = abs(test[j][0] - predicted[j][0])
        if(diff < rmse + eps) :
            Acc += 1
   
    Acc = float(Acc/len(predicted))
   
    print(f"The accuracy is {Acc} \n")
    print("The root mean squared error is {:.2f}.".format(rmse))
    print("The r2 score is {:.2f}.".format(r2))


    return(rmse)


def isUnderAttack(test, predicted, rmse, eps, thrs1, thrs2):
    isCompromised = False
    count = 0
    isSuspicious = 0

    #Redefine the test and predicted sets
    test = test.tolist()
    predicted = predicted.tolist()


    for j in range(0,len(predicted),4):
        diff = abs(np.subtract(test[j:j+4],predicted[j:j+4]))
        count = sum(diff > rmse + eps)[0]
        isSame = sum(ele == test[j] for ele in test[j:j+4])

        if(count >= thrs1 or isSame >= thrs1): # If statement will be come if(count >= thrs1 or isSame >= thrs1)
            isSuspicious+=1

        if(isSuspicious >= thrs2):
            isCompromised = True
            break
   
    return isCompromised

def computeMetrics(y_test, y_pred):
      
    #Compute True Positive (TP), False Positive (FP), False Negative (FN), and True Negative (TN) rates
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    
    for j in range(len(y_pred)):
      if(y_pred[j] ==1 and y_test[j] == 1):
        TP = TP + 1
      elif(y_pred[j] ==1 and y_test[j] == 0):
        FP = FP + 1
      elif(y_pred[j] ==0 and y_test[j] == 1):
        FN = FN + 1 
      elif(y_pred[j] ==0 and y_test[j] == 0):
        TN = TN + 1    

    
    #Compute DR, FPR, and HD
    DR = float(TP/(TP + FN))
    FPR = float(FP/(FP + TN))
    HD = abs(DR - FPR)
    
    return(DR, FPR, HD)




def RNN_forecastor(dataset, start_tind, end_tind, n_steps, my_path, filename):

    #Split up the dataset into training and testing
    training_set, val_set = train_test_split(dataset, start_tind, end_tind)
    sequence = len(val_set) #Set the amount that the rmse for loop should increment by


    training_set, val_set = training_set.T, val_set.T


    #Reshape and transforming the training dataset
    training_set = training_set.reshape(-1, 1).astype('float64')


    #We also need to split the training set into X_train (inputs) and y_train (outputs)
    #X_train will be a sequence based on a specific step size, 1-n_step, and the y_train will be the remainder of that
    #sequence 1-n:end
    X_train, y_train = split_sequence(training_set,n_steps)


    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)


    #Create the LSTM model and compile
    model_lstm = keras.models.Sequential()
    model_lstm.add(keras.layers.LSTM(units=215, activation="tanh", input_shape=(X_train.shape[1], 1)))
    model_lstm.add(keras.layers.Dropout(rate=0.2))
    model_lstm.add(keras.layers.Dense(units=1))

    #Plot Training and Validation loss
    #plot_trainingLoss(model_lstm, my_path, filename)


    model_lstm.compile(optimizer="RMSprop", loss="mse")


    model_lstm.fit(X_train, y_train, shuffle= False, epochs=35, batch_size=350, validation_split=0.2)


    # Format test set so it is in the same format as the training data
    val_set = val_set.reshape(-1, 1).astype('float64')
    X_val, y_val = split_sequence(val_set, n_steps)


    # Reshape
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)


    # Make prediction
    predicted_results = model_lstm.predict(X_val).astype('float64')


    #Print out the RMSE
    rmse = []
    for i in range(0,len(y_val), sequence):
      curRMSE = np.sqrt(mean_squared_error(y_val[i:i+sequence], predicted_results[i:i+sequence]))
      rmse.append(curRMSE)

    #AvgTrainingRMSE = evaluateTraining(y_val, predicted_results)


    #Return the model's predictions
    return (model_lstm, rmse)
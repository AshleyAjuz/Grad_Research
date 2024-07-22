import numpy as np
import pandas as pd
import random
from Functions.RNN_Forecastor import RNN_forecastor
from Functions.RNN_Forecastor import split_sequence
from Functions.RNN_Forecastor import isUnderAttack
from Functions.RNN_Forecastor import computeMetrics
from Functions.attackFunctions import attackFunctions
from SupplementalWork.preProcess import oversample
from Functions.evalRNN_Forecastor import evalRNN_Forecastor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#Dataset of energy readings 
benign_data = pd.read_csv("Output/Tidy_LCL_Data_2Y.csv").to_numpy().transpose()[1:,:]

#Need to create a label vector to denote that the readings are benign (y=0)
y_labs_B = [0 for i in range(len(benign_data))]

#Store an array for customer IDs in the benign dataset
custIds_benign = [x for x in range(len(benign_data))]

csv_path = "./Output/"

length, width = np.shape(benign_data)


#Determine split for training and testing
idx = round(0.7*width)

training_data = benign_data.T[0:idx]
testing_data = benign_data.T[idx+1:]

# Initialize values for RNN Forecastor
tstart = 0
tend = round(0.8*len(training_data)) # Spliting the training dataset into training and validation set
n_steps = 6 #This is the step size that the GRU will be given for it to then predict the next value


#Perform RNN
mdl, rmse = RNN_forecastor(training_data, tstart, tend, n_steps, csv_path, "PredictionsVsReal_NoClusters.png")

#Split up testing data into validation and final evaluation set
test_idx_val = round(0.7*len(testing_data))
test_val_data = testing_data[0:test_idx_val].T
final_eval_data= testing_data[test_idx_val+1:].T


#Evaluate the Validation dataset
thrs1 = 3 # If customer's data deviates more than three times in a month, this customer will be categorized as suspicious
thrs2 = 5 #Determines if the customer is malicious 
evalRNN_Forecastor(thrs1, thrs2, rmse, n_steps, mdl, custIds_benign, y_labs_B, test_val_data)


#Evaluate the final test dataset
evalRNN_Forecastor(thrs1, thrs2, rmse, n_steps, mdl, custIds_benign, y_labs_B, final_eval_data)
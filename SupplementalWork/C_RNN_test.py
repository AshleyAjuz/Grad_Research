import pandas as pd
import random
from Functions.attackFunctions import attackFunctions
from SupplementalWork.C_RNN import evalC_RNN
from SupplementalWork.preProcess import oversample


benign_data = pd.read_csv("Output/Tidy_LCL_Data.csv").to_numpy().transpose()[1:,:]

#Need to create a label vector to denote that the readings are benign (y=0)
y_labs_B = [0 for i in range(len(benign_data))]

#Create attack data using the benign datset
csv_path = "./Output/"
attack_data = attackFunctions(benign_data, 1, csv_path)

#Need to create a label vector to denote that the readings are malicious (y=1)
y_labs_M =  [1 for i in range(len(attack_data))]

#Combine the benign_data and attack_data together to create the complete dataset (referred to as X_c in reference)
X_c = benign_data.tolist() + attack_data

#Create new overall y_labs vector
y_labs = y_labs_B + y_labs_M


# Oversampling and Normalization
X_c, y_labs = oversample(X_c, y_labs)


##Shuffle the datasets to make sure training set includes malicious users
zipped = list(zip(X_c, y_labs))
random.shuffle(zipped)
X_c, y_labs = zip(*zipped)

#Training dataset
idx = round(0.8*len(X_c))
X_train = X_c[0:idx]
y_train = y_labs[0:idx]


#Testing dataset
X_test = X_c[idx:]
y_test = y_labs[idx:]


# Initialize values for C-RNN alg
I = 10 # num of epochs
B = 350 # Batch size for training
N = 64 # num of neurons for each layer

#Perform RNN
DR_values, FA_values, HD_values = evalC_RNN(X_train, y_train, X_test, y_test, N, B, I)

#Print out returned values
print(f"The DR_value is {DR_values}. The FA_value is {FA_values}. The HD_values is {HD_values}")
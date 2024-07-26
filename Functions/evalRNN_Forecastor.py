import numpy as np
import random
from sklearn.metrics import mean_squared_error,r2_score
from Functions.attackFunctions import attackFunctions
from Functions.RNN_Forecastor import split_sequence
from Functions.RNN_Forecastor import isUnderAttack
from Functions.RNN_Forecastor import computeMetrics

def evalRNN_Forecastor(thrs1, thrs2, rmse, n_steps, mdl, custIds_benign, y_labs_B, testData):
  
  #Create attack data using the benign datset
  attack_data = attackFunctions(testData, 1, "./Output/")
  
  #Need to create a label vector to denote that the readings are malicious (y=1)
  y_labs_M =  [1 for i in range(len(attack_data))]
  
  #Create customer ID array for the attack dataset
  custIDs_attack = custIds_benign * 5
  
  #Combine both sets of customer ID arrays
  custIDs = custIds_benign + custIDs_attack
  
  
  #Combine the benign_data and attack_data together to create the complete dataset (referred to as X_c in reference)
  X_c = (testData).tolist() + attack_data
  
  #Create new overall y_labs vector
  y_labs = y_labs_B + y_labs_M
  
  ##Shuffle the datasets 
  zipped = list(zip(X_c, y_labs, custIDs))
  random.shuffle(zipped)
  X_c, y_labs, custIDs = zip(*zipped)
   
  overall_res = []

  
  for i in range(len(X_c)):
      #Get the honest data for the current meter i
      honest_data = testData[custIDs[i],:]
      cur_data = np.array(X_c[i][:])
  
      #Format honest data by reshaping and splitting into the input and output subsets
      honest_data = honest_data.reshape(-1, 1).astype('float64')
  
      honest_data_X, honest_data_y = split_sequence(honest_data, n_steps)
  
      honest_data_X = honest_data_X.reshape(honest_data_X.shape[0],honest_data_X.shape[1],1)

      #Retrieve the model's predictions for current meter i based on the honest data
      predictions = mdl.predict(honest_data_X).astype('float64')
      curRMSE = rmse[custIDs[i]]
      #Set threshold1 (eps) equal to meter i's RMSE that was calculated during training * .40
      eps = (curRMSE * .40)
      
      cur_data = cur_data.reshape(-1,1).astype('float64')
  
      cur_data_X, cur_data_y = split_sequence(cur_data, n_steps)
      
      #check if current data X_c[i] is malicious or benign
      cur_res = isUnderAttack(cur_data_y,predictions, curRMSE, eps, thrs1, thrs2)
  
      #If meter is malicious, label as 1. If meter is benign, label as 0
      if cur_res:
          overall_res.append(1)
          
      else:
          overall_res.append(0)
          
  
  #Get accuracy of results by comparing the entries in the y_labels with the overall_results
  Acc = sum(np.array(overall_res) == np.array(y_labs))/len(y_labs)
  
  #Get remaining metrics form compute metrics function
  DR, FPR, HD = computeMetrics(y_labs, overall_res)
  
  
  #Print out returned values
  print(f"The overall Accuracy is {Acc*100}. The DR_value is {DR*100}. The FPR_value is {FPR*100}. The HD_values is {HD*100}. R2 is {r2_val} ")
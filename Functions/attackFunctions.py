import csv
import numpy as np
from random import seed
from random import randint
import pandas as pd 
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#Create the manipulated data for each smart meter
def attackFunctions(benign_readings, attackfreq, csv_path):


    #Create variable to store the number of meters present
    dimensions = np.shape(benign_readings)
   
    if(len(dimensions) == 1):
        numMeters = 1
        timeStamps = dimensions[0]
    else:
        numMeters, timeStamps= dimensions

    #Create lists for each attack
    A1 = []
    A2 = []
    A3 = []
    A4 = []
    A5 = []
    


    attackF = round(attackfreq * timeStamps)

    #Create array to randomly determine if the attack will occur in the current timestep
    attackOccur = [1 for i in range(attackF)]
    noAttack = [0 for i in range(timeStamps-attackF)]

    #Combine both arrays and randomly shuffle the order
    overall_Attack_Freq = noAttack + attackOccur
    random.shuffle(overall_Attack_Freq)

    #Conduct first attack (reduce energy readings by some fraction a)
    a = random.uniform(0.1, 0.6)

    #Initialize the result list
    res = [0 for i in range(timeStamps)]
    for x in range(numMeters):
        curMeter= benign_readings[x].tolist() if numMeters>1 else benign_readings.tolist()
        for y in range(timeStamps):
            if overall_Attack_Freq[y]==1:
                res[y] = a * curMeter[y]
            else:
                res[y] = curMeter[y]
        A1.append( res.copy() )

    #Conduct second attack (reduce energy readings by a dynamic amount)
    B = np.random.uniform(0.1, 0.6, timeStamps).tolist()

    for x in range(numMeters):
        #If statement to account for the case that only one customer is sent int
        curMeter= benign_readings[x].tolist() if numMeters>1 else benign_readings.tolist()
        for y in range(timeStamps):
            if overall_Attack_Freq[y]==1:
                res[y] = B[y] * curMeter[y]
            else:
                res[y] = curMeter[y]
        A2.append( res.copy() )

    #Conduct third attack (report energy reading of 0 during a certain interval)
    #If the interval of time needs to change for each customer, add this section of code (line 40-46) to below for loop
    t_i = np.random.randint(0, timeStamps, 1)[0]
    t_f = np.random.randint(t_i, timeStamps, 1)[0]
 

    #Create list of 1's
    list_multiplier = [1 for i in range(timeStamps)]

    list_multiplier[t_i:t_f] = [0 for x in range(t_f-t_i)]

    for x in range(numMeters):
        curMeter = benign_readings[x].tolist() if numMeters>1 else benign_readings.tolist()
        res = [list_multiplier[i] * curMeter[i] for i in range(len(curMeter))]
        A3.append( res.copy() )
 

    #Conduct fourth attack (report the mean of the energy readings )
    for x in range(numMeters):
        curMeter = benign_readings[x].tolist() if numMeters>1 else benign_readings.tolist()
        mean = sum(curMeter)/len(curMeter)
        for y in range(timeStamps):
            if overall_Attack_Freq[y] == 1:
                res[y] = mean
            else:
                res[y] = curMeter[y]
        A4.append( res.copy() )
       
    

    #Conduct fifth attack (report the mean of the energy readings that has been reduced by a dynamic amount )

    for x in range(numMeters):
        curMeter = benign_readings[x].tolist() if numMeters>1 else benign_readings.tolist()
        mean = sum(curMeter)/len(curMeter)
        for y in range(timeStamps):
            if overall_Attack_Freq[y] == 1:
                res[y] = B[y] * mean
            else:
                res[y] = curMeter[y]
        A5.append( res.copy() )
    

    #Concatenate all of the attack lists
    Overall_Attack = A1 + A2 + A3 + A4 + A5
   

    #Transform data into a csv file
    #df = pd.DataFrame(Overall_Attack)
    #df.to_csv(csv_path + "AttackData.csv")

    return(Overall_Attack)




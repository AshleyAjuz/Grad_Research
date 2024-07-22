import os
import pandas as pd 
import numpy as np  


#Function for reading the time series data from the ciruit's energy meters (Second Attempt)
# And storing results in a csv file and a list object

def GetMeterOutput(dss, csv_path):
    
    #Create variable to store the number of hours we are evaluating
    N = 24

    #Create a profile matrix that will store all meter voltage values at those timesteps
    profile = np.zeros((N,len(dss.meters_all_names())))

    #Store the names of all circuit meters
    names_meter = dss.meters_all_names()

    #Evaluate the circuit at each hour
    for i in range(N):
        #Solve the circuit
        dss.text("solve")
        
        #Start with first meter
        dss.meters_first()

        #Initialize count to 0
        count = 0

        #Loop through all of the meters
        for meter in names_meter:
            profile[i, count] = dss.meters_register_values()[0] # The energy output kWh of that register
            count += 1
            dss.meters_next() #move onto the next meter '''
    
    #Go to parent directory
    os.chdir('../../')

    
    doesExist = os.path.exists(csv_path)
    if not doesExist:
       # Create a new directory if path does not exist
        os.makedirs(csv_path)

    #Transform data into a csv file
    df = pd.DataFrame(profile)
    df.to_csv(csv_path + "MeterData.csv")

    #The following line transposes the matrix so that the cols become the rows and vice versa if needed
    #df = pd.DataFrame(total_energy_kw_list).T

    return(profile)

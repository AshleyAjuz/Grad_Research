import os
import pathlib
from OpenDSS_Work.placeMeters import placeMeters
from SupplementalWork.plotCustData import plotMeterOutputs
from OpenDSS_Work.GetMeterOutput import GetMeterOutput
from Functions.attackFunctions import attackFunctions
from OpenDSS_Work.change_loadProfiles import change_loadProfiles
from SupplementalWork.clusterData import FindOptimalClusters
from SupplementalWork.clusterData import clustData
from Functions.RNN_Forecastor import RNN_forecastor
from OpenDSS_Work.make_line import make_line
from itertools import chain
import numpy as np
import pandas as pd
import py_dss_interface


#Create the path for which the dss circuit is located
script_path = os.path.dirname(os.path.abspath(__file__)) 

dss = py_dss_interface.DSSDLL()

dss_file = pathlib.Path(script_path).joinpath("Feeders", "37Bus","ieee37.dss")

dss.text(f"compile [{dss_file}]")


#Place smart meters depending on desired configuration
MeterPlacements = ["l13","l33","l10","l21", "l18", "l16"]
TrackedLoads = ["load.s722c","load.s728", "load.s712c","load.s740c","load.s735c", "load.s731b"]

for i in range(len(TrackedLoads)):
    dss=make_line(dss,TrackedLoads[i])

dss = placeMeters(dss,MeterPlacements)

#Import typical household loadprofiles to assign to each load in the feeder
HouseholdLoadProfile = pd.read_csv("../../InputData/HouseholdLoadData.csv")
names_loads = dss.loads_all_names()

#Import EV loadprofiles
EVProfileData = pd.read_csv("../../InputData/DOE_EVData.csv").iloc[0,1:].tolist()
#EVProfileData =[]

#Create list of loads to have EVS
EVs = ["s722c","s728", "s712c","s740c","s735c", "s731b"]
dss = change_loadProfiles(dss,HouseholdLoadProfile,names_loads, EVs, EVProfileData) 

#dss.text('Plot Loadshape Object=LDs722c')

#Read smart meter data and store in an array
csv_path = "./Output/"
voltageProfiles = GetMeterOutput(dss, csv_path)

#Plot the Meter Data
plotMeterOutputs(voltageProfiles, dss.meters_all_names(), csv_path)


#Find optimal cluster amount and store plot in Output folder
transposed_VoltProf = np.transpose(voltageProfiles)
clustersFound = FindOptimalClusters(transposed_VoltProf, len(MeterPlacements), csv_path)

#Cluster data given the optimal number of clusters (found using the plot results from FindOptimalClusters)
optimal_clusters = 2
clusters = clustData(transposed_VoltProf, optimal_clusters)


#Create a collection of csv files from the clusters data to store data that are in the same clusters together
csvsCreated = create_csv(clusters,csv_path) 

#RNN algorithm
RNNModels = [0 for i in range(len(clusters)-1)] # Will store the models for each identified energy profile

#Defining parameter constants for RNN
start = 0
n_steps = 12

for i in range(len(RNNModels)):
    #Initialize dataset (take the average of the points in the cluster to create one representative energy profile )
    #dataset = list(chain.from_iterable(clusters[i][:]))
    dataset = np.array(clusters[i]).mean(axis=0).tolist()

    #Augment the data from 1 day to 1 week
    dataset = dataset*7

    #Change end index to ensure 80% of the data in the dataset is for training
    end = round(0.8*len(dataset))

    #Create filename for each RNN model
    filename = f"PredictionsVsReal_Plot_Model{i}.png"

    RNNModels[i] = RNN_forecastor(dataset,start,end,n_steps,csv_path, filename)



'''
Create attack data for each of the csv files created for each cluster

for i in range (len(clusters)):
    input_csv_file = f"Output/Group{i}.csv"
    output_csv_file =f"./Output/Group{i}_Attack_Data.csv"
    attackFunctions(input_csv_file, output_csv_file)

'''

print("Finished programing")



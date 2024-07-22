import py_dss_interface
import numpy as np
import pandas as pd

#Given a dss circuit and data (a 2D matrix containing label names and values)
#will instantiate loadshapes and apply them to the loads specified by 
#loadNames.
def change_loadProfiles(dss, data, loadNames, EVloads, EVdata):
    
    #Variable for storing the number of Loads
    num_loads= len(loadNames)

    #Create variable for count
    count = 0

    #Create the loadshapes
    for i in loadNames:
        if i in EVloads:
            ldShp = [data.iloc[count,1:].tolist()[x] + EVdata[x] for x in range(len(EVdata))]
            dss.text(f"New LoadShape.LD{loadNames[count]} npts=24 interval=1 mult={ldShp}")
        else:
            dss.text(f"New LoadShape.LD{loadNames[count]} npts=24 interval=1 mult={data.iloc[count,1:].tolist()}")
        count += 1

    #Assign the loadshapes to the user specified loads
    for i in loadNames:
        dss.loads_write_name(f"{i}")
        dss.loads_write_daily(f"LD{i}")
    
    return(dss)

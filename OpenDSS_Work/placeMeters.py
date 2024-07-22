import py_dss_interface
import pathlib

#Function will setup dss interface and install energy meters are user specified locations
def placeMeters(dss, Elements):
    
    #Get the number of meters
    num_meters = len(Elements)

    #Create the meters
    for i in range(num_meters):
        dss.text(f'New EnergyMeter.{Elements[i]}')
        dss.text(f"edit EnergyMeter.{Elements[i]} element=Line.{Elements[i]}_new enabled=True ")

    dss.solution_write_mode(1)
    dss.solution_write_number(1)

    #return the dss interface
    return dss


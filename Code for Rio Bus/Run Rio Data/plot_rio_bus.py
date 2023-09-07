import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import numpy as np
import pandas as pd

busData = pd.read_csv('treatedBusDataOnlyRoute.csv')
sf = gpd.read_file('data/33MUE250GC_SIR.shp')

for n in [20, 50, 100, 250]:
    temp_dat1 = busData[((busData.line == 383)) & 
                        (busData.date == '01-25-2019') &
                        (busData.time > '09:00:00') & 
                        (busData.time < '17:00:00') &
                        (busData.order == 'D13249')]
    # temp_dat2 = busData[((busData.line == 371)) & 
    #                     (busData.date == '01-25-2019') & 
    #                     (busData.time > '09:00:00') & 
    #                     (busData.time < '17:00:00') &
    #                     (busData.order == 'C51623')]
    # temp_dat3 = busData[((busData.line == 2381)) & 
    #                     (busData.time > '09:00:00') & 
    #                     (busData.time < '17:00:00') &
    #                     (busData.date == '01-25-2019') & 
    #                     (busData.order == 'D86706')]
    # temp_dat4 = busData[((busData.line == 935)) & 
    #                     (busData.time > '09:00:00') & 
    #                     (busData.time < '17:00:00') &
    #                     (busData.date == '01-25-2019') &
    #                     (busData.order == 'B28514')]
    rio = sf[sf.ID == 1535]
    bus_plot = rio.plot()
    
    bus_plot.xaxis.set_major_locator(MultipleLocator(0.1))
    bus_plot.yaxis.set_major_locator(MultipleLocator(0.1))
    
    # Change minor ticks to show every 5. (20/4 = 5)
    bus_plot.xaxis.set_minor_locator(MultipleLocator(0.005))
    bus_plot.yaxis.set_minor_locator(MultipleLocator(0.005))

    bus_plot.plot(temp_dat1.longitude, temp_dat1.latitude, 
             ".", markersize=0.2,color='black')
    # bus_plot.plot(temp_dat2.longitude, temp_dat2.latitude, 
    #          ".", markersize=0.2,color='red')
    # bus_plot.plot(temp_dat3.longitude, temp_dat3.latitude, 
    #          ".", markersize=0.2,color='purple')
    # bus_plot.plot(temp_dat4.longitude, temp_dat4.latitude, 
    #          ".", markersize=0.2,color='yellow')
    bus_plot.set_axisbelow(False)
    bus_plot.grid(which = 'major', linewidth=0.1)
    bus_plot.grid(which = 'minor', linewidth=0.1)
    bus_plot.set_title("Trajectory of four buses on 01/25/2019 (9am-5pm)")
    bus_plot.figure.savefig('bus_plot1.pdf')
    
    
    

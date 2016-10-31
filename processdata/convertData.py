import os
import numpy as np
import pandas as pd
import math
from math import *

from geographiclib.geodesic import Geodesic





        
posfilename = '../data/May_2016_nodes.csv'
outfilename = '../data/nodes_meters.csv'

posDF = pd.read_csv(posfilename) 

posDF=posDF[posDF['Name']!='Possible_wildebeest']

# convert from GPS coordinates to cartesian coordinates
startLon = np.mean(posDF['X'].values)#posDF['XCOORD'][0]
startLat = np.mean(posDF['Y'].values)#posDF['YCOORD'][0]

posDF['xm']=0
posDF['ym']=0
posDF['angle']=math.nan
lastid=-1
for i,pos in posDF.iterrows():
    if pos['Name']=='Known_direction':
        if pos['Pair_ID']==lastid:
            continue
        lastid=pos['Pair_ID']


    if i%1000==0:
        print(i,len(posDF))
    diff = Geodesic.WGS84.Inverse(startLat,startLon,pos['Y'],pos['X'])
    distance = diff['s12']
    angle = math.radians(90-diff['azi1']) # lat lon goes clockwise from north
    posDF.loc[i,'xm'] = distance*math.cos(angle)
    posDF.loc[i,'ym'] = distance*math.sin(angle)
    if pos['Name']=='Known_direction':
        diffNodes = Geodesic.WGS84.Inverse(posDF.loc[i+1,'Y'],posDF.loc[i+1,'X'],pos['Y'],pos['X'])
        posDF.loc[i,'angle'] = math.radians(90-diffNodes['azi1']) # lat lon goes clockwise from north

posDF=posDF[(posDF['Name']!='Known_direction') | pd.notnull(posDF['angle'])]
posDF.reset_index(drop=True,inplace=True)
posDF.to_csv(outfilename,index_col=False)

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
from math import *

from geographiclib.geodesic import Geodesic

posfilename = '../data/nodes_meters.csv'
outfilename = '../data/node_values.csv'


posDF = pd.read_csv(posfilename) 

KN=2



columns = ['id', 'dist_knn', 'angle_knn','align_knn']
df = pd.DataFrame(columns=columns) 

xvals=posDF['xm'].values
yvals=posDF['ym'].values
kdindex = posDF[posDF['Name']!='Known_direction'].index.values
count=0
for i,pos in posDF.iterrows():
    if pos['Name']!='Known_direction':
        continue
    

    # find nearest neighbour
    thisX=pos['xm']
    thisY=pos['ym']
    thisAngle=pos['angle']
    avDist=0 
    avCos=math.cos(2*thisAngle)
    avSin=math.sin(2*thisAngle)
    avAngle=0
    
    known_count=0
    all_count=0
    distances=(thisX-xvals)**2+(thisY-yvals)**2
    distances[i]=math.nan
    while known_count<KN:
        closest = np.nanargmin(distances)
        distances[closest]=math.nan
        # calculate distance and angle to nearest neighbour
        diff = Geodesic.WGS84.Inverse(pos['Y'],pos['X'],posDF['Y'][closest],posDF['X'][closest])

        if diff['s12']>50:
            break
        if all_count<KN:
            avDist += diff['s12']
            relAngle= math.radians(90-diff['azi1']) - pos['angle']
            avAngle += math.cos(2*relAngle) # lat lon goes clockwise from north

            all_count+=1
    
        if posDF['Name'][closest]=='Known_direction':
            nAngle = posDF['angle'][closest]
            avCos+=math.cos(2*nAngle)
            avSin+=math.sin(2*nAngle)
            known_count+=1


    if all_count<KN:
        continue
    avDist = avDist/float(all_count)
    avAngle = avAngle/float(all_count)
    align = ((avCos/float(known_count+1))**2 + (avSin/float(known_count+1))**2)**0.5
    if abs(align-0.33)<0.01:
        break

    df.loc[len(df)] = [count,avDist, avAngle, align]
    #break

df.to_csv(outfilename,index_col=False)

#            
#rowCount = int(len(posDF))
#        
## convert to a numpy array
#allData =  np.empty((rowCount,4))
#i = 0
#for _ , pos in posDF.groupby('FID'):
#    allData[i,0] = 0.5*(pos['xm'].iloc[0]+pos['xm'].iloc[1])
#    allData[i,1] = 0.5*(pos['ym'].iloc[0]+pos['ym'].iloc[1])
#    allData[i,2] = math.atan2(pos['ym'].iloc[0]-pos['ym'].iloc[1],pos['xm'].iloc[0]-pos['xm'].iloc[1])
#    allData[i,3] = pos['FID'].iloc[0]
#    i = i + 1
#    # we don't know heads from tails so add an entry for each direction
#    allData[i,0] = 0.5*(pos['xm'].iloc[0]+pos['xm'].iloc[1])
#    allData[i,1] = 0.5*(pos['ym'].iloc[0]+pos['ym'].iloc[1])
#    allData[i,2] = math.atan2(pos['ym'].iloc[1]-pos['ym'].iloc[0],pos['xm'].iloc[1]-pos['xm'].iloc[0])
#    allData[i,3] = pos['FID'].iloc[0]
#    i = i + 1
#
###checks
##s1=10
##s2=20
##plt.figure()
##plt.plot(posDF['XCOORD'].values[s1:s2],posDF['YCOORD'].values[s1:s2],'.')
##plt.figure()
##plt.quiver(allData[s1:s2,0],allData[s1:s2,1], np.cos(allData[s1:s2,2]), np.sin(allData[s1:s2,2]))
#
#
#
## build an array to store the relative angles and distances to all neighbours
#locations = np.zeros((0,3)).astype(np.float32) 
#for thisRow in range(rowCount):
#    thisX = allData[thisRow,0]
#    thisY = allData[thisRow,1]
#    thisAngle = (allData[thisRow,2])
#    thisTrack = (allData[thisRow,3])
#    
#    # find all animals at this time point in the clip that aren't the focal individual
#    window = allData[(allData[:,3]!=thisTrack),:]
#    rowLoc = np.zeros((0,3)).astype(np.float32) 
#    for w in window:
#        xj = w[0]
#        yj = w[1]
#        jAngle = (w[2])
#        r = ((((thisX-xj)**2+(thisY-yj)**2))**0.5)
#        if r>100:
#            continue
#        dx = xj - thisX
#        dy = yj - thisY
#        angle = math.atan2(dy,dx)
#        angle = angle - thisAngle
#        jAngle = jAngle - thisAngle
#        #angle = math.atan2(dy,dx)
#        theta = math.atan2(math.sin(angle), math.cos(angle))
#        jHeading  = math.atan2(math.sin(jAngle), math.cos(jAngle))
#        rowLoc = np.vstack((rowLoc,[r, theta, jHeading]))
#    locations = np.vstack((locations,rowLoc))
#
#
### POLAR PLOT OF RELATIVE POSITIONS
##BL = is approx 32 pixels
#binn2=19 # distance bins
#binn1=36
#
#dr = 0.5 # width of distance bins
#sr = 0.25 # start point of distance
#maxr=sr+(dr*binn2)
#theta2 = np.linspace(0.0,2.0 * np.pi, binn1+1)
#r2 = np.linspace(sr, maxr, binn2+1)
#areas = pi*((r2+dr)**2-r2**2)/binn1
#areas = areas[0:-1]
#areas=np.tile(areas,(binn1,1)).T
#
## wrap to [0, 2pi]
#locations[locations[:,1]<0,1] = locations[locations[:,1]<0,1] + 2 *pi
#
#hista2=np.histogram2d(x=locations[:,0],y=locations[:,1],bins=[r2,theta2],normed=1)[0]  
#
#hista2 =hista2/areas
#
#size = 8
## make a square figure
#
#fig1=plt.figure(figsize=(8,8))
#ax2=plt.subplot(projection="polar",frameon=False)
#im=ax2.pcolormesh(theta2,r2,hista2,lw=0.0,vmin=0,vmax=0.15,cmap='viridis')
##im=ax2.pcolormesh(theta2,r2,hista2,lw=0.0,vmin=0.0005,vmax=0.002,cmap='viridis')
#ax2.yaxis.set_visible(False)
#
## angle lines
#ax2.set_thetagrids(angles=np.arange(0,360,45),labels=['', '45°', '90°', '135°', '', '225°','270°', '315°'],frac=1.1)
#ax1 = ax2.figure.add_axes(ax2.get_position(), projection='polar',label='twin', frame_on=False,theta_direction=ax2.get_theta_direction(), theta_offset=ax2.get_theta_offset())
#ax1.yaxis.set_visible(False)
#ax1.set_thetagrids(angles=np.arange(0,360,45),labels=['front', '', '',  '', 'back', '','', ''],frac=1.1)
##colourbar
#position=fig1.add_axes([1.1,0.12,0.04,0.8])
#cbar=plt.colorbar(im,cax=position) 
#cbar.set_label('Neighbour density', rotation=90,fontsize='xx-large',labelpad=15)      
#
##body length legend - draws the ticks and 
#axes=ax2            
#factor = 0.98
#d = axes.get_yticks()[-1] #* factor
#r_tick_labels = [0] + axes.get_yticks()
#r_ticks = (np.array(r_tick_labels) ** 2 + d ** 2) ** 0.5
#theta_ticks = np.arcsin(d / r_ticks) + np.pi / 2
#r_axlabel = (np.mean(r_tick_labels) ** 2 + d ** 2) ** 0.5
#theta_axlabel = np.arcsin(d / r_axlabel) + np.pi / 2
#
## fixed offsets in x
#offset_spine = transforms.ScaledTranslation(-100, 0, axes.transScale)
#offset_ticklabels = transforms.ScaledTranslation(-10, 0, axes.transScale)
#offset_axlabel = transforms.ScaledTranslation(-40, 0, axes.transScale)
#
## apply these to the data coordinates of the line/ticks
#trans_spine = axes.transData + offset_spine
#trans_ticklabels = trans_spine + offset_ticklabels
#trans_axlabel = trans_spine + offset_axlabel
#axes.plot(theta_ticks, r_ticks, '-_k', transform=trans_spine, clip_on=False)
#
## plot the 'tick labels'
#for ii in range(len(r_ticks)):
#    axes.text(theta_ticks[ii], r_ticks[ii], "%.0f" % r_tick_labels[ii], ha="right", va="center", clip_on=False, transform=trans_ticklabels)
#
## plot the 'axis label'
#axes.text(theta_axlabel, r_axlabel, 'distance (meters)',rotation='vertical', fontsize='xx-large', ha='right', va='center', clip_on=False, transform=trans_axlabel)#             family='Trebuchet MS')
#
#
#fig1.savefig(posfilename + ".png",bbox_inches='tight',dpi=100)
## plot the 'spine'
#plt.figure()
#binn2=39 # distance bins
#
#
#dr = 0.5 # width of distance bins
#sr = 0.25 # start point of distance
#maxr=sr+(dr*binn2)
#
#r2 = np.linspace(sr, maxr, binn2+1)
#areas = pi*((r2+dr)**2-r2**2)
#areas = areas[0:-1]
#
#
#
#hista1=np.histogram(locations[:,0],bins=r2,normed=1)[0]  
#
#hista1 =hista1/areas
#plt.plot(r2[:-1]+0.5*dr,hista1,'.-')
#plt.ylim([0,0.01])
#plt.savefig(posfilename + "_dist.png",bbox_inches='tight',dpi=100)
#
##       
##      
#### POLAR PLOT OF ALIGNMENT
##cosRelativeAngles = np.cos(locations[:,2])
##sinRelativeAngles = np.sin(locations[:,2])
##
### find the average cos and sin of the relative headings to calculate circular statistics
##histcos=binned_statistic_2d(x=locations[:,0],y=locations[:,1],values=cosRelativeAngles, statistic='mean', bins=[r2,theta2])[0]  
##histsin=binned_statistic_2d(x=locations[:,0],y=locations[:,1],values=sinRelativeAngles, statistic='mean', bins=[r2,theta2])[0]  
##
### mean is atan and std dev is 1-R
##relativeAngles = np.arctan2(histsin,histcos)
##stdRelativeAngles = np.sqrt( 1 - np.sqrt(histcos**2+histsin**2))
##minSD = np.nanmin(stdRelativeAngles)
##maxSD = np.nanmax(stdRelativeAngles)
##
##stdRelativeAngles[np.isnan(stdRelativeAngles)]=0
##
##
##fig1=plt.figure(figsize=(8,8))
##ax2=plt.subplot(projection="polar",frameon=False)
##im=ax2.pcolormesh(theta2,r2,stdRelativeAngles,lw=0.0,vmin=minSD,vmax=maxSD,cmap='viridis_r')
##ax2.yaxis.set_visible(False)
##
### angle lines
##ax2.set_thetagrids(angles=np.arange(0,360,45),labels=['', '45°', '90°', '135°', '', '225°','270°', '315°'],frac=1.1)
##ax1 = ax2.figure.add_axes(ax2.get_position(), projection='polar',label='twin', frame_on=False,theta_direction=ax2.get_theta_direction(), theta_offset=ax2.get_theta_offset())
##ax1.yaxis.set_visible(False)
##ax1.set_thetagrids(angles=np.arange(0,360,45),labels=['front', '', '',  '', 'back', '','', ''],frac=1.1)
###colourbar
##position=fig1.add_axes([1.1,0.12,0.04,0.8])
##cbar=plt.colorbar(im,cax=position) 
##cbar.set_label('Circular variance', rotation=90,fontsize='xx-large',labelpad=15)      
##
###body length legend - draws the ticks and 
##axes=ax2            
##factor = 0.98
##d = axes.get_yticks()[-1] #* factor
##r_tick_labels = [0] + axes.get_yticks()
##r_ticks = (np.array(r_tick_labels) ** 2 + d ** 2) ** 0.5
##theta_ticks = np.arcsin(d / r_ticks) + np.pi / 2
##r_axlabel = (np.mean(r_tick_labels) ** 2 + d ** 2) ** 0.5
##theta_axlabel = np.arcsin(d / r_axlabel) + np.pi / 2
##
### fixed offsets in x
##offset_spine = transforms.ScaledTranslation(-100, 0, axes.transScale)
##offset_ticklabels = transforms.ScaledTranslation(-10, 0, axes.transScale)
##offset_axlabel = transforms.ScaledTranslation(-40, 0, axes.transScale)
##
### apply these to the data coordinates of the line/ticks
##trans_spine = axes.transData + offset_spine
##trans_ticklabels = trans_spine + offset_ticklabels
##trans_axlabel = trans_spine + offset_axlabel
##axes.plot(theta_ticks, r_ticks, '-_k', transform=trans_spine, clip_on=False)
##
### plot the 'tick labels'
##for ii in range(len(r_ticks)):
##    axes.text(theta_ticks[ii], r_ticks[ii], "%.0f" % r_tick_labels[ii], ha="right", va="center", clip_on=False, transform=trans_ticklabels)
##
### plot the 'axis label'
##axes.text(theta_axlabel, r_axlabel, 'metres',rotation='vertical', fontsize='xx-large', ha='right', va='center', clip_on=False, transform=trans_axlabel)#             family='Trebuchet MS')
##
##
##fig1.savefig("order.png",bbox_inches='tight',dpi=100)
##
##
#### POLAR PLOT OF ATTRACTION
##
##
### find the average cos and sin of the relative headings to calculate circular statistics
##histcos=binned_statistic_2d(x=locations[:,0],y=locations[:,1],values=cosRelativeAngles, statistic='mean', bins=[r2,theta2])[0]  
##histsin=binned_statistic_2d(x=locations[:,0],y=locations[:,1],values=sinRelativeAngles, statistic='mean', bins=[r2,theta2])[0]  
##
##
##angles = 0.5*(theta2[0:-1]+theta2[1:])
##angles=np.tile(angles,(binn2,1))
##
##toOrigin = -(histcos*np.cos(angles) + histsin*np.sin(angles))
##fig1=plt.figure(figsize=(8,8))
##ax2=plt.subplot(projection="polar",frameon=False)
##im=ax2.pcolormesh(theta2,r2,toOrigin,lw=0.0,vmin=np.nanmin(toOrigin),vmax=np.nanmax(toOrigin),cmap='viridis')
##ax2.yaxis.set_visible(False)
##
### angle lines
##ax2.set_thetagrids(angles=np.arange(0,360,45),labels=['', '45°', '90°', '135°', '', '225°','270°', '315°'],frac=1.1)
##ax1 = ax2.figure.add_axes(ax2.get_position(), projection='polar',label='twin', frame_on=False,theta_direction=ax2.get_theta_direction(), theta_offset=ax2.get_theta_offset())
##ax1.yaxis.set_visible(False)
##ax1.set_thetagrids(angles=np.arange(0,360,45),labels=['front', '', '',  '', 'back', '','', ''],frac=1.1)
###colourbar
##position=fig1.add_axes([1.1,0.12,0.04,0.8])
##cbar=plt.colorbar(im,cax=position) 
##cbar.set_label('Attraction', rotation=90,fontsize='xx-large',labelpad=15)      
##
###body length legend - draws the ticks and 
##axes=ax2            
##factor = 0.98
##d = axes.get_yticks()[-1] #* factor
##r_tick_labels = [0] + axes.get_yticks()
##r_ticks = (np.array(r_tick_labels) ** 2 + d ** 2) ** 0.5
##theta_ticks = np.arcsin(d / r_ticks) + np.pi / 2
##r_axlabel = (np.mean(r_tick_labels) ** 2 + d ** 2) ** 0.5
##theta_axlabel = np.arcsin(d / r_axlabel) + np.pi / 2
##
### fixed offsets in x
##offset_spine = transforms.ScaledTranslation(-100, 0, axes.transScale)
##offset_ticklabels = transforms.ScaledTranslation(-10, 0, axes.transScale)
##offset_axlabel = transforms.ScaledTranslation(-40, 0, axes.transScale)
##
### apply these to the data coordinates of the line/ticks
##trans_spine = axes.transData + offset_spine
##trans_ticklabels = trans_spine + offset_ticklabels
##trans_axlabel = trans_spine + offset_axlabel
##axes.plot(theta_ticks, r_ticks, '-_k', transform=trans_spine, clip_on=False)
##
### plot the 'tick labels'
##for ii in range(len(r_ticks)):
##    axes.text(theta_ticks[ii], r_ticks[ii], "%.0f" % r_tick_labels[ii], ha="right", va="center", clip_on=False, transform=trans_ticklabels)
##
### plot the 'axis label'
##axes.text(theta_axlabel, r_axlabel, 'metres',rotation='vertical', fontsize='xx-large', ha='right', va='center', clip_on=False, transform=trans_axlabel)#             family='Trebuchet MS')
##
##
##fig1.savefig("toOrigin.png",bbox_inches='tight',dpi=100)
##

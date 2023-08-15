import matplotlib.pyplot as plt
from matplotlib import dates
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable

from IPython.display import HTML
%config InlineBackend.figure_format ='retina'

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os
import json
import requests

import scipy as sp
from scipy import signal

from pathlib import Path

from sunpy.net import Fido
from sunpy.net import attrs as a

from tsmoothie.smoother import *

from astral.sun import sun
from astral import LocationInfo

#This makes a list of usable files from the specificed location and joins them. Shown below.
path = 'Folder path'

files = Path(path).glob('*.csv')

groups = list()
for i in (files):
    group = pd.read_csv(i, skiprows=16, delimiter=',', usecols=['DHO'], skipinitialspace=True, names=['datetimes', 'DHO'])
    groups.append(group)

group = pd.concat(groups, 'columns', ignore_index=True)

#Reads in data from given csv and adds a column with mean data
DataRead = pd.read_csv('file location.csv', skiprows=16, delimiter=',',  skipinitialspace=True, names=['datetimes', 'DHO'])

frames = list()
frames.append(Birr)
frames.append(group.mean(axis=1))
prep = pd.concat(frames, 'columns', ignore_index=True)

#Makes the pandas df with the deviation from the mean
final = list()
final.append(prep[0])
final.append((20*np.log10(prep[1]))-(20*np.log10(prep[2])))
AvgDev = pd.concat(final, 'columns', ignore_index=True)

# Converting the VLF time data to datetime object
# Create a new list
time_dho = []
# For each time value in the datetimes column of the vlf data, convert it to 
# a datetime object and append it to the new list (time_naa)
for i in AvgDev[0]:
    time_dho.append(datetime.strptime(i, '%Y-%m-%d %H:%M:%S'))

#Clean vlf data by replacing infinite values with nans (makes for better plotting)
vlf_fix_dB = (AvgDev[1])
vlf_fix_dB[vlf_fix_dB == -np.inf] = np.nan
vlf_fix_dB[vlf_fix_dB == +np.inf] = np.nan

#Smooth data for ease of interpreting
gaussian = GaussianSmoother(n_knots=300, sigma=0.00001)
gaussian.smooth(vlf_fix_dB)
vlf_filtered_dB = gaussian.smooth_data[0]

#Additional interval for possible noise detection
low, up = gaussian.get_intervals('prediction_interval')

#Combining filtering info into one dataframe
upper = pd.DataFrame(up[0], columns=['up'])
lower = pd.DataFrame(low[0], columns=['low'])
filtered = pd.DataFrame(vlf_filtered_dB, columns=['data'])
time = pd.DataFrame(time_dho, columns=['time'])

noisecheck = list()
noisecheck.append(filtered)
noisecheck.append(upper)
noisecheck.append(lower)
Noise = pd.concat(noisecheck, 'columns', ignore_index=True)

# Plot figure of mean data to check for abnormalities (Optional)
# Create a figure
fig = plt.figure(figsize=(12,4))
ax = plt.gca()

# Plot the mean data
ax.plot(time_dho, 20*np.log10(prep[2]), color='k', lw=1, ls='-', label='VLF DHO Average', alpha=1)

# Set the x label to be the current date in the first time object 
ax.set_xlabel(str(time_dho[0].strftime("%d/%m/%Y"))+ '[UTC]', fontsize=14)
# Set y-value to string below.
ax.set_ylabel('VLF Signal [dB]', fontsize=14)

# Set x-axis limit to be start time with a window of 1 day
ax.set_xlim(time_dho[0], time_dho[0]+timedelta(days=1))
# Set y-axis limit to be 80 dB to the max recorded dB in the data +5 dB
ax.set_ylim(int(np.min(20*np.log10(prep[2]))-5), int(np.max(20*np.log10(prep[2]))+5))

# Make the x-axis appear in time format (Hours:mins)
date_format = mdates.DateFormatter('%H:%M')
ax.xaxis_date()
ax.xaxis.set_major_formatter(date_format)

ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(bottom=True, top=True, left=True, right=True,direction="in")

# plot a grid
plt.grid(alpha=0.4, ls='dotted')

plt.show()

#Converts the filtered data back to a dataframe and pairs it back with time.
filtered_dB = pd.DataFrame(vlf_filtered_dB, columns=['Filtered'])

reports = list()
reports.append(AvgDev[0])
reports.append(filtered_dB)
reportlist = pd.concat(reports, 'columns', ignore_index=True)

#Using scipy signal.find_peaks and assigning it to variables.
peakamp = []
peaktime = []

peaktime = reportlist[0][(signal.find_peaks(reportlist[1], 10, 0.00001, (10*(60/2)))[0])]

peakamp= reportlist[1][(signal.find_peaks(reportlist[1], 10, 0.00001, (10*(60/2)))[0])]

#Converts to DataFrame for ease of use. Not a requirement but quality of life.
listObj = [peaktime,peakamp]
ser = pd.DataFrame(listObj)

# Get GOES flares for the given date:
event_type = "FL"
tstart = "2023/05/20"
tend = "2023/05/21"
result = Fido.search(a.Time(tstart, tend), a.hek.EventType(event_type), a.hek.FL.GOESCls > "C6.0", a.hek.OBS.Observatory == "GOES")
hek_results = result["hek"]
filtered_results = hek_results["event_starttime", "event_peaktime", "event_endtime", "fl_goescls", "ar_noaanum"]
filtered_results[:5]

#Creates list for each flare class and separates them
cClass = []
mClass = []
xClass = []
for event in filtered_results:
    if str(event["fl_goescls"]).__contains__('C'):
        cClass.append(event)
    if str(event["fl_goescls"]).__contains__('M'):
        mClass.append(event)
    if str(event["fl_goescls"]).__contains__('X'):
        xClass.append(event)
    else:
        pass
    
def get_sun_info(date):
    # Getting sunrise, sunset data etc.
    city = LocationInfo("Birr", "Ireland", "Europe", 53.3871, -6.3375)
    s = sun(city.observer, date= date)
    return s

# Create a figure
fig = plt.figure(figsize=(12,4))

ax = plt.gca()

# Plot the raw data (looks quite noisy)
ax.plot(time_dho, vlf_filtered_dB, color='k', lw=1, ls='-', label='VLF DHO Adjusted', alpha=1)

# Set the x label to be the current date in the first time object 
ax.set_xlabel(str(time_dho[0].strftime("%d/%m/%Y"))+ '[UTC]', fontsize=12)
# Set y-value to string below.
ax.set_ylabel('VLF Signal Deviation from Mean [dB]', fontsize=12)

# Set x-axis limit to be start time with a window of 1 day
ax.set_xlim(time_dho[0]+timedelta(hours=0), time_dho[0]+timedelta(hours=24))
# Set y-axis limit to be 80 dB to the max recorded dB in the data +5 dB
ax.set_ylim(int(np.min(vlf_filtered_dB)-5), int(np.max(vlf_filtered_dB)+5))

# Make the x-axis appear in time format (Hours:mins)
date_format = mdates.DateFormatter('%H:%M')
ax.xaxis_date()
ax.xaxis.set_major_formatter(date_format)

ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(bottom=True, top=True, left=True, right=True,direction="in")

#Plots peaks onto graph
for val in (ser):
    time = datetime.strptime(ser[val][0], '%Y-%m-%d %H:%M:%S')
    plt.plot(time, ser[val][1], '.b', ms=10)
    
#Plots GOES flares by class
for flare in cClass:
    plt.vlines(flare['event_peaktime'].datetime,ymin=int(np.min(vlf_filtered_dB)-5), ymax=int(np.max(vlf_filtered_dB)+5), color='green', alpha=0.3)
    plt.text(flare['event_peaktime'].datetime-timedelta(minutes=25),int(np.min(vlf_filtered_dB)),flare['fl_goescls'], c='g', rotation=90)
    
for flare in mClass:
    plt.vlines(flare['event_peaktime'].datetime,ymin=int(np.min(vlf_filtered_dB)-5), ymax=int(np.max(vlf_filtered_dB)+5), color='orange', alpha=0.3)
    plt.text(flare['event_peaktime'].datetime-timedelta(minutes=25),int(np.min(vlf_filtered_dB)),flare['fl_goescls'], c='orange', rotation=90)    
    
for flare in xClass:
    plt.vlines(flare['event_peaktime'].datetime,ymin=int(np.min(vlf_filtered_dB)-5), ymax=int(np.max(vlf_filtered_dB)+5), color='red', alpha=0.3)
    plt.text(flare['event_peaktime'].datetime-timedelta(minutes=25),int(np.min(vlf_filtered_dB)),flare['fl_goescls'], c='r', rotation=90)
    
#Adds black bars at sides to sunrise/sunset times
s = get_sun_info(time_dho[0])    
plt.fill_betweenx([int(np.min(vlf_filtered_dB)-5), int(np.max(vlf_filtered_dB)+5)],time_dho[0],(s["sunrise"]), color='k', alpha=0.7, hatch='//')
plt.fill_betweenx([int(np.min(vlf_filtered_dB)-5), int(np.max(vlf_filtered_dB)+5)],(s["sunset"]),time_dho[-1], color='k', alpha=0.7, hatch='//')

#Adds the up and low segment of tsmoothie filtering
plt.fill_between(time_dho, Noise[1], Noise[2], alpha=0.4)
 
# plot a grid
plt.grid(alpha=0.4, ls='dotted')

plt.show()

#Prints peaks time and intensity
for val in (ser):
    print('There was a peak detected at ' + str(ser[val][0]) + ' with a deviation of ' + str(ser[val][1]) + ' dB from the mean.')
# -*- coding: utf-8 -*-
"""Weather API.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1c9Pvnz0FP1UZuHERFprTGOaDXV4vfh-4
"""

#needed to make web requests
import requests
#store the data we get as a dataframe
import pandas as pd
#convert the response as a strcuctured json
import json
#mathematical operations on lists
import numpy as np
#parse the datetimes we get from NOAA
from datetime import datetime
#add the access token you got from NOAA
Token = 'XnvxAnqwtzCGLaepxKsuVihUwMhZkCti'
#Long Beach Airport station
station_id = 'GHCND:USW00094846'

#initialize lists to store data - Average tenperature (TAVG)
dates_temp = []
dates_prcp = []
temps = []
prcp = []
#for each year from 2015-2018 ...
for year in range(2015, 2019):
    year = str(year)
    print('working on year '+year)
    #make the api call
    r = requests.get('https://www.ncdc.noaa.gov/cdo-web/api/v2/data?datasetid=GHCND
	&datatypeid=TAVG&limit=1000
	&stationid=GHCND:USW00023129&startdate='+year+'-01-01
	&enddate='+year+'-12-31', headers={'token':Token})
    #load the api response as a json
    d = json.loads(r.text)
    #get all items in the response which are average temperature readings
    avg_temps = [item for item in d['results'] if item['datatype']=='TAVG']
    #get the date field from all average temperature readings
    dates_temp += [item['date'] for item in avg_temps]
    #get the actual average temperature from all average temperature readings
    temps += [item['value'] for item in avg_temps]
#initialize dataframe
df_temp = pd.DataFrame()
#populate date and average temperature fields (cast string date to datetime and convert temperature from tenths of Celsius to Fahrenheit)
df_temp['date'] = [datetime.strptime(d, "%Y-%m-%dT%H:%M:%S") for d in dates_temp]
df_temp['avgTemp'] = [float(v)/10.0*1.8 + 32 for v in temps]



#initialize lists to store data - Precipitation (PRCP)
dates_temp = []
dates_prcp = []
temps = []
prcp = []
#for each year from 2015-2018 ...
for year in range(2015, 2019):
    year = str(year)
    print('working on year '+year)
    #make the api call
    r = requests.get('https://www.ncdc.noaa.gov/cdo-web/api/v2/data?datasetid=GHCND
	&datatypeid=PRCP&limit=1000&stationid=GHCND:USW00023129
	&startdate='+year+'-01-01&enddate='+year+'-12-31', headers={'token':Token})
    #load the api response as a json
    d = json.loads(r.text)
    #get all items in the response which are average temperature readings
    avg_temps = [item for item in d['results'] if item['datatype']=='PRCP']
    #get the date field from all average temperature readings
    dates_temp += [item['date'] for item in avg_temps]
    #get the actual average temperature from all average temperature readings
    temps += [item['value'] for item in avg_temps]
#initialize dataframe
df_prcp = pd.DataFrame()
#populate date and average temperature fields (cast string date to datetime and convert temperature from tenths of Celsius to Fahrenheit)
df_prcp['date'] = [datetime.strptime(d, "%Y-%m-%dT%H:%M:%S") for d in dates_temp]
df_prcp['prcp'] = [float(v)/10.0*1.8 + 32 for v in temps]

#initialize lists to store data - Average Wind Speed (AWND)
dates_temp = []
dates_prcp = []
temps = []
prcp = []
#for each year from 2015-2018 ...
for year in range(2015, 2019):
    year = str(year)
    print('working on year '+year)
    #make the api call
    r = requests.get('https://www.ncdc.noaa.gov/cdo-web/api/v2/data?datasetid=GHCND
	&datatypeid=AWND&limit=1000&stationid=GHCND:USW00023129
	&startdate='+year+'-01-01&enddate='+year+'-12-31', headers={'token':Token}) 
    #load the api response as a json
    d = json.loads(r.text)
    #get all items in the response which are average temperature readings
    avg_temps = [item for item in d['results'] if item['datatype']=='AWND']
    #get the date field from all average temperature readings
    dates_temp += [item['date'] for item in avg_temps]
    #get the actual average temperature from all average temperature readings
    temps += [item['value'] for item in avg_temps]
#initialize dataframe
df_wind = pd.DataFrame()
#populate date and average temperature fields (cast string date to datetime and convert temperature from tenths of Celsius to Fahrenheit)
df_wind['date'] = [datetime.strptime(d, "%Y-%m-%dT%H:%M:%S") for d in dates_temp]
df_wind['wind'] = [float(v)/10.0*1.8 + 32 for v in temps]

#Merge all weather scrapes
print(df_temp.shape)
print(df_prcp.shape)
print(df_wind.shape)
df_prcp.head()
weather = df_temp.merge(df_prcp,on=['date'])
weather = weather.merge(df_wind,on=['date'],how='left')
weather.head()
#Split date column to year, month and day
weather['date'] = pd.to_datetime(weather['date'])
weather['Month'] = weather['date'].dt.month
weather['Day'] = weather['date'].dt.day
weather['Year'] = weather['date'].dt.year
weather = weather.drop(columns=['date'])
weather.to_csv('C:\Data Analytics\Sem 3\Dataset\LocationData\Distances\weatherapi.csv',index=False)
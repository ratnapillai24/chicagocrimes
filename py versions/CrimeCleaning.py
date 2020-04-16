
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import censusgeocode as cg
import censusbatchgeocoder
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; 
warnings.filterwarnings('ignore')


# In[2]:


#Read Crime Data
crimedata = pd.read_csv("C:\\Data Analytics\\Sem 3\\Dataset\\CrimeDataset.csv")


# In[3]:


crime = crimedata.copy() #Make a copy of crime data
#filter crime data 2014-2019
newcrime = crime[(crime['Year']>2014) & (crime['Year']<2019)] 
newcrime.shape #1069147,31


# In[4]:


newcrime = newcrime.drop(columns=['Unnamed: 0']) #Remove the column Unnamed: 0
newcrime.columns
newcrime.head()


# In[5]:


#Data Cleaning of Crime Data
#Standardise Column Names
newcrime.columns = ['ID','Case_Number','Reported_Date', 'Block', 'IUCR','Primary_Type','Description','Location_Description','Arrest','Domestic','Beat','District','Ward','Community_Area','FBI_Code','X_Coordinate','Y_Coordinate','Year','Updated_On','Latitude','Longitude','Location','Historical_Wards_2003to2015','Zip_Codes','Community_Areas','Census_Tracts','Wards','Boundaries_ZIP_codes','Police_Districts','Police_Beats']
newcrime.columns
#Missing values check
newcrime.isna().sum() #has null values


# In[6]:


newcrime['Reported_Date'] = pd.to_datetime(newcrime['Reported_Date'])


# In[7]:


newcrime['Month'] = newcrime['Reported_Date'].dt.month
newcrime['Day'] = newcrime['Reported_Date'].dt.day
newcrime['Weekday'] = newcrime['Reported_Date'].dt.dayofweek
newcrime['HourOfDay'] = newcrime['Reported_Date'].dt.hour


# In[9]:


narcotics = newcrime[(newcrime.Primary_Type == 'NARCOTICS') | (newcrime.Primary_Type == 'LIQUOR LAW VIOLATION') | (newcrime.Primary_Type == 'OTHER NARCOTIC VIOLATION') | (newcrime.Primary_Type == 'PUBLIC PEACE VIOLATION') | (newcrime.Primary_Type == 'WEAPONS VIOLATION') | (newcrime.Primary_Type == 'PUBLIC INDECENCY') | (newcrime.Primary_Type == 'ASSAULT') | (newcrime.Primary_Type == 'HOMICIDE') | (newcrime.Primary_Type == 'GAMBLING')  ]


# In[10]:


narcotics.Year.value_counts()


# In[14]:


sns.set(font_scale=2)
msno.heatmap(narcotics)


# In[15]:


#Statistics of arrest and non arrest ()
arrest_count = narcotics['Arrest'].value_counts().sort_index()
totalcrimes = narcotics['ID'].count()
print(arrest_count)
arrestdf = arrest_count.rename_axis('unique_values').reset_index(name='counts')
print (arrestdf)
nonarrestper = round((arrestdf.iloc[0,1]/totalcrimes.astype(int))*100,2)
arrestper = round((arrestdf.iloc[1,1].astype(int)/totalcrimes)*100,2)
print(arrestper)
print(nonarrestper)


# In[24]:


fig, ax = plt.subplots()
sns.set(font_scale=1.5)
color_palette_list = ['#009ACD','#ADD8E6','#63D1F4', '#0EBFE9',   
                      '#C1F0F6', '#0099CC']
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#565656'
plt.rcParams['axes.labelcolor']= '#565656'
plt.rcParams['xtick.color'] = '#565656'
plt.rcParams['ytick.color'] = '#565656'
plt.rcParams['font.size']=15

labels = ['Arrest Percent', 
         'Non-Arrest Percent']

percentages = [arrestper, nonarrestper]
explode=(0.1,0)
ax.pie(percentages, explode=explode, labels=labels,  
       colors=color_palette_list[0:2], autopct='%1.0f%%', 
       shadow=False, startangle=0,   
       pctdistance=1.2,labeldistance=1.4)
ax.axis('equal')
ax.legend(frameon=False, bbox_to_anchor=(1.5,0.8))


# In[26]:


locdata = pd.read_csv('C:/Data Analytics/Sem 3/Dataset/LocationsZip.csv') #load the fetched latitude & longitude using censusgeocode


# In[28]:


locdata['Block'] = locdata['PropertyAddress'].str.split(',').str[0]
locdata['ZipCode'] = locdata.ZipCode.astype(int)
locdata['ZipCode'] = locdata.ZipCode.astype(str)


# In[29]:


#Lookup for zipcode
narcotics.Zip_Codes.replace('NaN', np.NaN, inplace=True)
narcotics.loc[narcotics['Zip_Codes'].isnull(),'Zip_Codes'] = narcotics['Block'].map(locdata.ZipCode)
#Lookup for latitude
narcotics.loc[narcotics['Latitude'].isna(),'Latitude'] = narcotics['Block'].map(locdata.LAT)
#Lookup for longitude
narcotics.loc[narcotics['Longitude'].isna(),'Longitude'] = narcotics['Block'].map(locdata.LON)


# In[30]:


#Fill missing Zipcodes
s = locdata.set_index('Block')['ZipCode']
narcotics['Zip_Codes'] = narcotics['Zip_Codes'].fillna(narcotics['Block'].map(s))
#Fill missing latitude
s1 = locdata.set_index('Block')['LAT']
narcotics['Latitude'] = narcotics['Latitude'].fillna(narcotics['Block'].map(s1))
#Fill missing longitude
s2 = locdata.set_index('Block')['LON']
narcotics['Longitude'] = narcotics['Longitude'].fillna(narcotics['Block'].map(s2))
#Fill Location Description with Unknown
narcotics['Location_Description'] = narcotics['Location_Description'].fillna("Unknown")
#Fill Location
Locationcon = '(' + locdata['LAT'].astype(str) + ',' + ' ' + locdata['LON'].astype(str) + ')'
locdata['Location'] = Locationcon
s3 = locdata.set_index('Block')['Location']
narcotics['Location'] = narcotics['Location'].fillna(narcotics['Block'].map(s3))


# In[31]:


narcotics.isna().sum()


# In[32]:


#Drop columns derived from location attributes to eliminate noise and do not contribute to the predictor variable
#Since we have longitude and latitude, there is no requirement of x-coordinate and y co-ordinate
narcotics = narcotics.drop(columns=['X_Coordinate','Y_Coordinate','Historical_Wards_2003to2015','Census_Tracts','Wards','Boundaries_ZIP_codes','Police_Districts','Police_Beats','District','Ward','Community_Area','Community_Areas','Beat'])
narcotics.isna().sum()


# In[35]:


narcotics.to_csv('C:\\Data Analytics\\Sem 3\\ICT Solution\\Data Sets\\CleanCrimes.csv',index=False)


# In[36]:


narcotics.shape #166100

#Extract only null rows for location & zip codes
locnull = narcotics[narcotics['Latitude'].isnull()]
locnull.shape
city = 'Chicago'
state = 'IL'
address = locnull['Block']
address = address.drop_duplicates()
addressdata = pd.DataFrame(address)
addressdata['city'] = city
addressdata['state'] = state
addressdata.columns = ['address','city','state']
addressdata = addressdata.reset_index()
addressdata['id'] = np.arange(1,len(addressdata)+1)
addressdata = addressdata.drop(columns=['index'])
#print (addressdata)
#Rearrange columns - addressdata
cols = list(addressdata.columns)
a, b = cols.index('address'), cols.index('id')
cols[b], cols[a] = cols[a], cols[b]
addressdata = addressdata[cols]
addressdata.head()
addressdata.to_csv('C:\\Data Analytics\\Sem 3\\Dataset\\addressData.csv',index=False)

# In[ ]:


import datetime
print(datetime.datetime.now())
addressdata.head()


# In[ ]:


addressdata.shape


# In[ ]:


fetchaddress = addressdata.to_dict("records")


# In[ ]:


print(datetime.datetime.now())
results = censusbatchgeocoder.geocode(fetchaddress.to_dict("records"),zipcode=None)
print(datetime.datetime.now())
#2019-10-18 20:07:03.927510
#2019-10-18 20:10:55.237334


# In[ ]:


pd_df = pd.DataFrame(results)


# In[ ]:


pd_df.to_csv('C:\\Data Analytics\\Sem 3\\ICT Solution\\Data Sets\\extractgeocodesdata.csv',index=False)


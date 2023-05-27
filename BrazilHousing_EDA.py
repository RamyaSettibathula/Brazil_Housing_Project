#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import statistics as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import norm
from scipy.stats import kurtosis
from tabulate import tabulate
import matplotlib as mat


# In[3]:


df = pd.read_csv('DS1_C5_S3_BazilHousing_Data_Hackathon.csv')
df


# # TASK-1 : PRE PROCESSING THE DATA

# In[3]:


df.isnull().sum()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


def seperate_data_types(df):
    categorical = []
    continous = []
    for column in df.columns:
        if df[column].dtypes == object:
            categorical.append(column)
        else:
            continous.append(column)
    return categorical,continous
categorical,continous = seperate_data_types(df)


from tabulate import tabulate
table = [categorical,continous]
print(tabulate({'Categorical': categorical,'Continous':continous}, headers =['categorical','continous']))


# In[7]:


def info_of_cat(col):
    print(f'Unique values in {col} are : {df[col].unique()}')
    print(f'mode of  {col} is : {df[col].mode()[0]}')
    print(f'total count of missing values in {col} is : {df[col].isnull().sum()}')


# In[8]:


def info_of_numerical(col):
    print(f'the mean of the {col} are : {df[col].mean()}')
    print(f'the median of the {col} are : {df[col].median()}')
    print(f'mode of  {col} is : {df[col].mode()[0]}')
    print(f' standard deviation of  {col} is : {df[col].std()}')
    print(f'total count of missing values in {col} is : {df[col].isnull().sum()}')


# In[9]:


info_of_cat('city')


# # TASK-2 : FIND OUT EXPENSIVE CITIES AND REMOVE FROM THE DATA

# ### most expensive cities visualization and then remove that expensive cities, start analysing the data

# In[10]:


fig,ax = plt.subplots(figsize=(10,5))
ax.set_title('Frequency of Cities')
sb.countplot(x= 'city' ,data = df)
for bar in ax.patches:
    value = f'{round(bar.get_height(),2)}'
    x = bar.get_x() + bar.get_width()/2 
    y= bar.get_height()  
    ax.annotate(value,(x,y), va = 'bottom',ha = 'center')
plt.show()


# # Interpretation:-From above graph sao paulo and Ria de janeiro city are more expensive. For our analysis we remove these two cities and continue the our analysis.

# In[4]:


df1 = df[(df['city']!='São Paulo') & (df['city']!='Rio de Janeiro') ]
df1


# # TASK-3 : CALCULATE CENTRAL TENDANCY CITYWISE ON NUMERICAL COLUMNS AND INTERPRETE IT.

# # central tendency city wise

# In[12]:


def ct(df,col1,col2):
    print(df.groupby(by=[col1])[[col2]].mean())
    print(df.groupby(by=[col1])[[col2]].median())
    print(pd.pivot_table(df, index=col1,values=col2, aggfunc=lambda x: x.mode()[0]))


# In[13]:


ct(df1,'city','total (R$)')


# # Interpretation:- there is difficult to plot a graph with total column because huge difference between min value and max value. so we do analysis on the basis of central tendency. by comparing these three city'Belo Horizonte' have highest values and 'Campinas' have lowest value. 

# In[14]:


ct(df1,'city','rent amount (R$)')


# # Interpretation:- citywise rental rate central tendency. Here also 'Campinas' is lowest. and 'Belo Horizonte' is highest median

# In[15]:


ct(df1,'city','property tax (R$)')


# # Interpretation:- citywise property tax central tendency. Here also 'Porto Alegre' is lowest. and 'Belo Horizonte' is highest median

# In[16]:


ct(df1,'city','hoa (R$)')


# # Interpretation:- citywise HOA central tendency. Here also 'Porto Alegre' is lowest. and 'Belo Horizonte' is highest median

# # TASK-4: CALCULATE MEASURES OF DISPERSION CITYWISE ON NUMERICAL COLUMNS AND INTERPRETE IT.

# # measures of dispersion citywise

# In[17]:


bh = df1[(df1['city']=='Belo Horizonte')]
cc= df1[(df1['city']=='Campinas')]
pa= df1[(df1['city']=='Porto Alegre')]


# In[18]:


def md(df,col):# function it calculates var,sd,cv,skewness,min,max,mean etc
    print('variance ',df[col].var())
    print('SD       ', df[col].std())
    print('CV       ', (df[col].std()/df[col].mean())*100)
    print('Skewness ',df[col].skew())
    print(df[col].describe())


# ### Belo Horizonte

# In[19]:


md(bh,'rent amount (R$)')


# In[20]:


md(bh,'area')


# In[21]:


md(bh,'total (R$)')


# ### Campinas

# In[22]:


md(cc,'rent amount (R$)')


# In[23]:


md(cc,'area')


# In[24]:


md(cc,'total (R$)')


# ### Porto Alegre

# In[25]:


md(pa,'rent amount (R$)')


# In[26]:


md(pa,'area')


# In[27]:


md(pa,'total (R$)')


# # INTERPRETATION: From measures of dispersion, i calculated area,rent amount, total amount with citywise. i used these values for filtering the data atlast. I observed here every mean,skewness,standard deviation,variance, coefficient of variation with citywise. Belo horizonto is highest in every parameter.Campinas, Porto Alegre cities are lowest in every parameters.

# # TASK-5 : FIND OUT CORRELATION ON EACH COLUMN WITH OTHER COLUMNS IN THE DATA AND INTERPRETE IT.

# # correlation

# In[28]:


fig,ax = plt.subplots(figsize = (10,7))
corr = df1.corr()
sb.heatmap(corr,annot= True)
plt.show()


# # INTERPRETATION:- This is a heatmap. It shows correlation of each column with other column in one go.By the color of the each box also indicates the correlation. With - symbol numbers are negatively correlated. Normal numbers are positively correlated. 1 is the strongly correlated.

# # TASK-6 : PERFORM UNIVARIATE ANALYSIS AND ANALYZE DATA AND INTERPRETE IT.

# # Univariate analysis

# In[29]:


fig, ax = plt.subplots(1,2, figsize = (15,7))
sb.countplot(x= df1['city'],ax = ax[0])
data = df1['city'].value_counts()
label = data.keys()
plt.pie(x = data,labels = label, autopct = '%0.2f%%',explode = [0.02,0.02,0.02], pctdistance = 0.5)
for bar in ax[0].patches:
    value = f'{round(bar.get_height(),2)}'
    x = bar.get_x() + bar.get_width()/2 
    y= bar.get_height()  
    ax[0].annotate(value,(x,y), va = 'bottom',ha = 'center')
    
ax[0].set_title('Frequency of Cities')
plt.title('Proportion of Cities')
plt.show()


# # INTERPRETATION:-  From above two graphs the frequency of Belo Horizonte city has highest frequency with 38%. Campinas has lowest frequency with 25%

# In[30]:


fig, ax = plt.subplots(1,2, figsize = (10,4))
sb.countplot(x= df1['animal'],ax = ax[0])
data = df1['animal'].value_counts()
label = data.keys()
plt.pie(x = data,labels = label, autopct = '%0.2f%%',explode = [0.02,0.02], pctdistance = 0.5)
for bar in ax[0].patches:
    value = f'{round(bar.get_height(),2)}'
    x = bar.get_x() + bar.get_width()/2 
    y= bar.get_height()  
    ax[0].annotate(value,(x,y), va = 'bottom',ha = 'center')

ax[0].set_title('Frequency of animals')
plt.title('Proportion of animals')
plt.show()


# # INTERPRETATION:- From above two graphs the frequency of accept animals has highest frequency with 80%. not acept animals has lowest frequency with 20%

# In[31]:


fig, ax = plt.subplots(1,2, figsize = (10,4))
sb.countplot(x= df1['furniture'],ax = ax[0])
data = df1['furniture'].value_counts()
label = data.keys()
plt.pie(x = data,labels = label, autopct = '%0.2f%%',explode = [0.02,0.02], pctdistance = 0.5)
for bar in ax[0].patches:
    value = f'{round(bar.get_height(),2)}'
    x = bar.get_x() + bar.get_width()/2 
    y= bar.get_height()  
    ax[0].annotate(value,(x,y), va = 'bottom',ha = 'center')

ax[0].set_title('Frequency of furniture')
plt.title('Proportion of furniture')
plt.show()


# # INTERPRETATION:- From above two graphs the frequency of not furniture has highest frequency with 81%. Furnished has lowest frequency with 19%

# # TASK-7 : PERFORM BI-VARIATE ANALYSIS AND ANALYZE DATA AND INTERPRETE IT.

# # bi-variate analysis

# categorical-categorical

# In[32]:


fig,ax = plt.subplots(figsize=(10,4))
ax.set_title('city vs animal')
sb.countplot(x= 'city',hue= 'animal', data = df1)
plt.show()


# # INTERPRETATION:- In three cities animals acceptation count is high. so if the family have animals , there is no restriction for keeping it.

# In[33]:


fig,ax = plt.subplots(figsize=(10,4))
ax.set_title('city vs furniture')
sb.countplot(x= 'city',hue= 'furniture', data = df1)
plt.show()


# # INTERPRETATION:- In three cities not furnished is high. So most of the families live without furnished. It is best for middle families and Bachelors.Porto Alegre is the city with highest furnished city.

# In[34]:


fig,ax = plt.subplots(1,2,figsize=(15,5))
ax[0].set_title('city vs rent')
ax[1].set_title('city vs rent')
bins = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000]
sb.histplot(x=df1['rent amount (R$)'], hue =df1['city'],ax=ax[0],bins=bins)
sb.boxplot(x=df1['city'], y =df1['rent amount (R$)'],ax=ax[1])
plt.show()


# # outlier analysis

# In[35]:


mean = int(df1['rent amount (R$)'].mean())
x = df1[(df1['rent amount (R$)']>5000)].index
print(x)
for index in x:
    df1.loc[index,'rent amount (R$)'] = mean


# In[36]:


fig,ax = plt.subplots(1,2,figsize=(15,5))
ax[0].set_title('city vs rent')
ax[1].set_title('city vs rent')
bins = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000]
sb.histplot(x=df1['rent amount (R$)'], hue =df1['city'],ax=ax[0],bins=bins)
sb.boxplot(x=df1['city'], y =df1['rent amount (R$)'],ax=ax[1])
plt.show()


# # INTERPRETATION:- After outlier analysis,city vs rent, the city 'Belo Horizoto ' have more spread that means it have high rental rates. 'Porto Alegre' have less spread means low rental rates from the box plot. Histogram is right skewed.

# In[54]:


fig,ax = plt.subplots(figsize=(8,4))
ax.set_title('city vs property tax')
bins = [0,50,100,150,200,250,300,350,400,450,500]
sb.histplot(x=df1['property tax (R$)'], hue =df1['city'],bins=bins)
plt.show()


# # INTERPRETATION:-Above Histogram is right skewed.Observing the histplot 0-100 property tax is in city 'porto alegre'. 100 above property tax is in the city 'Belo Horizonto'.In 'Campinas'city property tax is 400 nearly.

# In[38]:


fig,ax = plt.subplots(1,2,figsize=(10,4))
bins = [1,3,5,7,9,11]
ax[0].set_title('city vs rooms')
ax[1].set_title('city vs rooms')
sb.histplot(hue=df1['city'],x =df1['rooms'],ax=ax[0],bins = bins)
sb.boxplot(y=df1['rooms'], x =df1['city'],ax=ax[1])
plt.show()


# # INTERPRETATION:-Above histogram is right skewed. Box plots have very few outliers. 1,2 rooms available  in the city 'porto alegre',more than two rooms are available in the city 'Belo Horizonto'

# In[39]:


df2=df1.groupby(by=['city'])[['property tax (R$)']].mean()
df3=df1.groupby(by=['city'])[['area']].mean()
df4=df1.groupby(by=['city'])[['rent amount (R$)']].mean()
df5=df1.groupby(by=['city'])[['fire insurance (R$)']].mean()
df6=df1.groupby(by=['city'])[['total (R$)']].mean()
df7=df1.groupby(by=['city'])[['hoa (R$)']].mean()


# In[40]:


fig, ax = plt.subplots(2,3, figsize=(12,8))
d2 = df2['property tax (R$)']
ax[0,0].set_title('city vs property tax')
label = df2.index
ax[0,0].pie(x = d2,labels = label, autopct = '%0.2f%%',explode = [0.02,0.02,0.02], pctdistance = 0.5)
d2 = df3['area']
ax[0,1].set_title('city vs area')
label = df3.index
ax[0,1].pie(x = d2,labels = label, autopct = '%0.2f%%',explode = [0.02,0.02,0.02], pctdistance = 0.5)
d2 = df4['rent amount (R$)']
ax[0,2].set_title('city vs rent amount (R$)')
label = df4.index
ax[0,2].pie(x = d2,labels = label, autopct = '%0.2f%%',explode = [0.02,0.02,0.02], pctdistance = 0.5)
d2 = df5['fire insurance (R$)']
ax[1,0].set_title('city vs fire insurance (R$)')
label = df5.index
ax[1,0].pie(x = d2,labels = label, autopct = '%0.2f%%',explode = [0.02,0.02,0.02], pctdistance = 0.5)
d2 = df6['total (R$)']
ax[1,1].set_title('city vs total (R$)')
label = df6.index
ax[1,1].pie(x = d2,labels = label, autopct = '%0.2f%%',explode = [0.02,0.02,0.02], pctdistance = 0.5)
d2 = df7['hoa (R$)']
ax[1,2].set_title('city vs hoa (R$)')
label = df7.index
ax[1,2].pie(x = d2,labels = label, autopct = '%0.2f%%',explode = [0.02,0.02,0.02], pctdistance = 0.5)
plt.show()


# # INTERPRETATION:-Above pie charts represents categorical- numerical columns, the city column with some important numerical columns. From observing all the above pie charts i concluded that the city 'Belo Horizonto' is apt for large families. The city 'Porto Alegre' is apt for bachelors, Remaining city 'Campinas' is apt for mid-families.

#    

# # TASK-8 : PERFORM MULTI-VARIATE ANALYSIS AND ANALYZE DATA AND INTERPRETE IT.

# # Multivariate analysis

# ## According to above analysis, i filtered some data for Bachelors, Mid-families and Large families then choose one city for relocating.

# # For bachelors

# In[41]:


bachelor=df1[(df1['area']<110)&(df1['bathroom']==1)&(df1['rooms']==1)&(df1['rent amount (R$)']<1500)&(df1['total (R$)']<2140)
             &(df1['property tax (R$)']<59)]


# In[42]:


bachelor


# In[43]:


fig, ax = plt.subplots(1,2, figsize = (10,4))
sb.countplot(x= bachelor['city'],ax = ax[0])
ax[0].set_title('Frequency of cities')
ax[1].set_title('Proportion of cities')
data = bachelor['city'].value_counts()
label = data.keys()
plt.pie(x = data,labels = label, autopct = '%0.2f%%',explode = [0.02,0.02,0.02], pctdistance = 0.5)
for bar in ax[0].patches:
    value = f'{round(bar.get_height(),2)}'
    x = bar.get_x() + bar.get_width()/2 
    y= bar.get_height()  
    ax[0].annotate(value,(x,y), va = 'bottom',ha = 'center')
plt.show()


# # FINAL INTERPRETATION:- From above bar chart and pie chart the city *'Porto Alegre'* is the best for relocating bachelors.

# # for middle families

# In[44]:


mf=df1[(df1['area']>110)&(df1['area']<207)&(df1['bathroom']>1)&(df1['bathroom']<=2)&(df1['rooms']>2)&(df1['rooms']<=3)
       &(df1['rent amount (R$)']<2364)]
mf


# In[45]:


fig, ax = plt.subplots(1,2, figsize = (10,4))
sb.countplot(x= mf['city'],ax = ax[0])
ax[0].set_title('Frequency of cities')
ax[1].set_title('Proportion of cities')
data = mf['city'].value_counts()
label = data.keys()
plt.pie(x = data,labels = label, autopct = '%0.2f%%',explode = [0.02,0.02,0.02], pctdistance = 0.5)
for bar in ax[0].patches:
    value = f'{round(bar.get_height(),2)}'
    x = bar.get_x() + bar.get_width()/2 
    y= bar.get_height()  
    ax[0].annotate(value,(x,y), va = 'bottom',ha = 'center')
plt.show()


# # INTERPRETATION:- From above bar chart and pie chhart the city *'Campinas'* is the best for relocating mid-sized families.

# # for large families

# In[46]:


largefam=df1[(df1['area']>207)&(df1['bathroom']>2)&(df1['rooms']>3)&(df1['total (R$)']>6351)&(df1['rent amount (R$)']>2364)
             &(df1['floor']==0)]
largefam


# In[47]:


fig, ax = plt.subplots(1,2, figsize = (10,4))
sb.countplot(x= largefam['city'],ax = ax[0])
ax[0].set_title('Frequency of cities')
ax[1].set_title('Proportion of cities')
data = largefam['city'].value_counts()
label = data.keys()
plt.pie(x = data,labels = label, autopct = '%0.2f%%',explode = [0.02,0.02,0.02], pctdistance = 0.5)
for bar in ax[0].patches:
    value = f'{round(bar.get_height(),2)}'
    x = bar.get_x() + bar.get_width()/2 
    y= bar.get_height()  
    ax[0].annotate(value,(x,y), va = 'bottom',ha = 'center')
plt.show()


# # INTERPRETATION:- From above bar chart and pie chhart the city *'Belo Horizonte'* is the best for relocating large families.

# In[16]:


fig,ax = plt.subplots(figsize=(14,6))
meltdf = pd.melt(df1, id_vars = ['city'], value_vars = ['property tax (R$)','rent amount (R$)','total (R$)'],
                 var_name = 'variable',value_name = 'value')
sb.lineplot(x = 'city', hue = 'variable', y = 'value', data = meltdf  )


# In[17]:


df1.columns


# In[18]:


continous


# In[20]:


fig,ax = plt.subplots(figsize=(14,6))
meltdf = pd.melt(df1, id_vars = ['city'], value_vars = [
 'rooms',
 'bathroom',
 'parking spaces',
 'floor'], var_name = 'variable',value_name = 'value')
sb.lineplot(x = 'city', hue = 'variable', y = 'value', data = meltdf  )


# In[5]:


df1.columns


# In[6]:


fig,ax = plt.subplots(figsize=(7,5))
ax.set_title('hoa vs total ')
sb.regplot(x=df1['hoa (R$)'], y =df1['total (R$)'])
plt.show()


# In[9]:


fig,ax = plt.subplots(figsize=(7,5))
ax.set_title('fire insurance vs rent amount ')
sb.scatterplot(x=df1['rent amount (R$)'], y =df1['fire insurance (R$)'])
plt.show()


# In[ ]:





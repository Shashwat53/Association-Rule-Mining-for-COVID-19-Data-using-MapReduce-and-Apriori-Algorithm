#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from IPython.display import Image
sns.set(style="darkgrid", palette="pastel", color_codes=True)
sns.set_context("paper")
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "seaborn"
from plotly.subplots import make_subplots


# Load the COVID-19 dataset
data = pd.read_csv('worldometer_data.csv')


# In[8]:


# Get the top 10 countries with the highest number of total cases
top_10_total_cases = data.sort_values('TotalCases', ascending=False).head(10)

# Create a bar chart of the top 10 countries with the highest number of total cases
plt.figure(figsize=(12, 6))
plt.bar(top_10_total_cases['Country/Region'], top_10_total_cases['TotalCases'], color='green')
plt.title('Top 10 countries with the highest number of total cases')
plt.xlabel('Country')
plt.ylabel('Total cases')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[9]:


# Group the data by continent and sum the total cases, deaths, and recoveries
by_continent = data.groupby('Continent')['TotalCases', 'TotalDeaths', 'TotalRecovered'].sum()

# Create a stacked bar chart of the total cases, deaths, and recoveries by continent
by_continent.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Total cases, deaths, and recoveries by continent')
plt.xlabel('Continent')
plt.ylabel('Number of cases')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[10]:


df = pd.read_csv("covid_19_clean_complete.csv", parse_dates = ['Date'])
df.head()


# In[11]:


df.columns


# In[12]:


df.info()


# In[13]:


df.describe(include = 'object')


# In[14]:


a = df.Date.value_counts().sort_index()
print('The first date is:',a.index[0])
print('The last date is:',a.index[-1])


# In[8]:


df.isnull().sum()


# In[15]:


#Renaming the coulmns for easy usage
df.rename(columns={'Date': 'date', 
                     'Province/State':'state',
                     'Country/Region':'country',
                     'Lat':'lat', 'Long':'long',
                     'Confirmed': 'confirmed',
                     'Deaths':'deaths',
                     'Recovered':'recovered'
                    }, inplace=True)

# Active Case = confirmed - deaths - recovered
df['active'] = df['confirmed'] - df['deaths'] - df['recovered']


# In[10]:


df1 = df
df1['date'] = pd.to_datetime(df1['date'])
df1['date'] = df1['date'].dt.strftime('%m/%d/%Y')
df1 = df1.fillna('-')
fig = px.density_mapbox(df1, lat='lat', lon='long', z='confirmed', radius=20,zoom=1, hover_data=["country",'state',"confirmed"],
                        mapbox_style="carto-positron", animation_frame = 'date', range_color= [0, 10000],title='Spread of Covid-19')
fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
fig.show()


# In[16]:


top = df[df['date'] == df['date'].max()]
world = top.groupby('country')['confirmed','active','deaths'].sum().reset_index()
world.head()


# In[79]:


figure = px.choropleth(world, locations="country", 
                    locationmode='country names', color="active", 
                    hover_name="country", range_color=[1,500000], 
                    color_continuous_scale="Peach", 
                    title='Countries with Active Cases')
figure.show()


# In[17]:


fig = px.scatter_mapbox(top, lat="lat", lon="long", hover_name="country", hover_data=["country","recovered"],
                        color_discrete_sequence=["fuchsia"], zoom=0.5, height=300,title='Recovered count of each country' )
fig.update_layout(
    mapbox_style="white-bg",
    mapbox_layers=[
        {
            "below": 'traces',
            "sourcetype": "raster",
            "source": [
                "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
            ]
        }
      ])
fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
fig.show()


# In[81]:


world['size'] = world['deaths'].pow(0.2)
fig = px.scatter_geo(world, locations="country",locationmode='country names', color="deaths",
                     hover_name="country", size="size",hover_data = ['country','deaths'],
                     projection="natural earth",title='Death count of each country')
fig.show()


# In[18]:


plt.figure(figsize= (15,10))
plt.xticks(rotation = 90 ,fontsize = 10)
plt.yticks(fontsize = 15)
plt.xlabel("Dates",fontsize = 30)
plt.ylabel('Total cases',fontsize = 30)
plt.title("Worldwide Confirmed Cases Over Time" , fontsize = 30)
total_cases = df.groupby('date')['date', 'confirmed'].sum().reset_index()
total_cases['date'] = pd.to_datetime(total_cases['date'])


ax = sns.pointplot( x = total_cases.date.dt.week ,y = total_cases.confirmed , color = 'r')
ax.set(xlabel='Weeks', ylabel='Total cases')


# In[83]:


top = df[df['date'] == df['date'].max()]
top_casualities = top.groupby(by = 'country')['confirmed'].sum().sort_values(ascending = False).head(20).reset_index()
top_casualities


# In[19]:


plt.figure(figsize= (15,10))
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel("Total cases",fontsize = 30)
plt.ylabel('Country',fontsize = 30)
plt.title("Top 20 countries having most confirmed cases" , fontsize = 30)
ax = sns.barplot(x = top_casualities.confirmed, y = top_casualities.country)
for i, (value, name) in enumerate(zip(top_casualities.confirmed,top_casualities.country)):
    ax.text(value, i-.05, f'{value:,.0f}',  size=10, ha='left',  va='center')
ax.set(xlabel='Total cases', ylabel='Country')


# In[85]:


top_actives = top.groupby(by = 'country')['active'].sum().sort_values(ascending = False).head(20).reset_index()
top_actives


# In[86]:


plt.figure(figsize= (15,10))
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel("Total cases",fontsize = 30)
plt.ylabel('Country',fontsize = 30)
plt.title("Top 20 countries having most active cases" , fontsize = 30)
ax = sns.barplot(x = top_actives.active, y = top_actives.country)
for i, (value, name) in enumerate(zip(top_actives.active, top_actives.country)):
    ax.text(value, i-.05, f'{value:,.0f}',  size=10, ha='left',  va='center')
ax.set(xlabel='Total cases', ylabel='Country')


# In[87]:


top_deaths = top.groupby(by = 'country')['deaths'].sum().sort_values(ascending = False).head(20).reset_index()
top_deaths


# In[88]:


plt.figure(figsize= (15,10))
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel("Total cases",fontsize = 30)
plt.ylabel('Country',fontsize = 30)
plt.title("Top 20 countries having most deaths" , fontsize = 30)
ax = sns.barplot(x = top_deaths.deaths, y = top_deaths.country)
for i, (value, name) in enumerate(zip(top_deaths.deaths,top_deaths.country)):
    ax.text(value, i-.05, f'{value:,.0f}',  size=10, ha='left',  va='center')
ax.set(xlabel='Total cases', ylabel='Country')


# In[89]:


top_recovered = top.groupby(by = 'country')['recovered'].sum().sort_values(ascending = False).head(20).reset_index()
top_recovered


# In[90]:


plt.figure(figsize= (15,10))
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel("Total cases",fontsize = 30)
plt.ylabel('Country',fontsize = 30)
plt.title("Top 20 countries having most recovered cases" , fontsize = 30)
ax = sns.barplot(x = top_recovered.recovered, y = top_recovered.country)
for i, (value, name) in enumerate(zip(top_recovered.recovered,top_recovered.country)):
    ax.text(value, i-.05, f'{value:,.0f}',  size=10, ha='left',  va='center')
ax.set(xlabel='Total cases', ylabel='Country')


# In[91]:


rate = top.groupby(by = 'country')['recovered','confirmed','deaths'].sum().reset_index()
rate['recovery percentage'] =  round(((rate['recovered']) / (rate['confirmed'])) * 100 , 2)
rate['death percentage'] =  round(((rate['deaths']) / (rate['confirmed'])) * 100 , 2)
rate.head()


# In[37]:


mortality = rate.groupby(by = 'country')['death percentage'].sum().sort_values(ascending = False).head(20).reset_index()
mortality


# In[92]:


plt.figure(figsize= (15,10))
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel("Total cases",fontsize = 30)
plt.ylabel('Country',fontsize = 30)
plt.title("Top 20 countries having most mortality rate" , fontsize = 30)
ax = sns.barplot(x = mortality['death percentage'], y = mortality.country)
for i, (value, name) in enumerate(zip(mortality['death percentage'], mortality.country)):
    ax.text(value, i-.05, f'{value:,.0f}',  size=10, ha='left',  va='center')
ax.set(xlabel='Mortality Rate in percentage', ylabel='Country')


# In[39]:


recovery = rate.groupby(by = 'country')['recovery percentage'].sum().sort_values(ascending = False).head(20).reset_index()
recovery


# In[93]:


plt.figure(figsize= (15,10))
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel("Total cases",fontsize = 30)
plt.ylabel('Country',fontsize = 30)
plt.title("Top 20 countries having most recovery rate" , fontsize = 30)
ax = sns.barplot(x = recovery['recovery percentage'], y = recovery.country)
for i, (value, name) in enumerate(zip(recovery['recovery percentage'], recovery.country)):
    ax.text(value, i-.05, f'{value:,.0f}',  size=10, ha='left',  va='center')
ax.set(xlabel='Recovery Rate in percentage', ylabel='Country')


# In[96]:


china =  df[df.country == 'China']
china = china.groupby(by = 'date')['recovered', 'deaths', 'confirmed', 'active'].sum().reset_index()
china['date'] = china['date'].apply(lambda x : datetime.strptime(x, "%m/%d/%Y"))
china.head()


# In[95]:


us =  df[df.country == 'US']
us = us.groupby(by = 'date')['recovered', 'deaths', 'confirmed', 'active'].sum().reset_index()
us = us.iloc[33:].reset_index().drop('index', axis = 1)
us['date'] = us['date'].apply(lambda x : datetime.strptime(x, "%m/%d/%Y"))
us.head()


# In[97]:


italy =  df[df.country == 'Italy']
italy = italy.groupby(by = 'date')['recovered', 'deaths', 'confirmed', 'active'].sum().reset_index()
italy = italy.iloc[9:].reset_index().drop('index', axis = 1)
italy['date'] = italy['date'].apply(lambda x : datetime.strptime(x, "%m/%d/%Y"))
italy.head()


# In[98]:


india =  df[df.country == 'India']
india = india.groupby(by = 'date')['recovered', 'deaths', 'confirmed', 'active'].sum().reset_index()
india = india.iloc[8:].reset_index().drop('index', axis = 1)
india['date'] = india['date'].apply(lambda x : datetime.strptime(x, "%m/%d/%Y"))
india.tail()


# In[39]:


plt.figure(figsize=(15,30))
a = plt.subplot(4, 1, 1)
sns.pointplot(china.date.dt.week ,china.confirmed)
plt.title("China's Confirmed Cases Over Time" , fontsize = 25)
plt.ylabel('Total cases', fontsize = 15)
plt.xlabel('No. of Weeks', fontsize = 15)

plt.subplot(4, 1, 2)
sns.pointplot(us.date.dt.week ,us.confirmed)
plt.title("US's Confirmed Cases Over Time" , fontsize = 25)
plt.xlabel('No. of Weeks', fontsize = 15)
plt.ylabel('Total cases', fontsize = 15)

plt.subplot(4, 1, 3)
sns.pointplot(italy.date.dt.week ,italy.confirmed)
plt.title("Italy's Confirmed Cases Over Time" , fontsize = 25)
plt.xlabel('No. of Weeks', fontsize = 15)
plt.ylabel('Total cases', fontsize = 15)

plt.subplot(4, 1, 4)
sns.pointplot(india.date.dt.week ,india.confirmed)
plt.title("India's Confirmed Cases Over Time" , fontsize = 25)
plt.xlabel('No. of Weeks', fontsize = 15)
plt.ylabel('Total cases', fontsize = 15)

plt.subplots_adjust(bottom=0.01, top=0.9)


# In[40]:


plt.figure(figsize=(15,30))
plt.subplot(4, 1, 1)
sns.pointplot(china.index ,china.active, color = 'r')
plt.title("China's Active Cases Over Time" , fontsize = 25)
plt.xlabel('No. of Days', fontsize = 15)
plt.ylabel('Total cases', fontsize = 15)

plt.subplot(4, 1, 2)
sns.pointplot(us.index ,us.active, color = 'r')
plt.title("US's Active Cases Over Time" , fontsize = 25)
plt.xlabel('No. of Days', fontsize = 15)
plt.ylabel('Total cases', fontsize = 15)

plt.subplot(4, 1, 3)
sns.pointplot(italy.index ,italy.active, color = 'r')
plt.title("Italy's Active Cases Over Time" , fontsize = 25)
plt.xlabel('No. of Days', fontsize = 15)
plt.ylabel('Total cases', fontsize = 15)

plt.subplot(4, 1, 4)
sns.pointplot(india.index ,india.active, color = 'r')
plt.title("India's Active Cases Over Time" , fontsize = 25)
plt.xlabel('No. of Days', fontsize = 15)
plt.ylabel('Total cases', fontsize = 15)

plt.subplots_adjust(bottom=0.01, top=0.9)


# In[41]:


plt.figure(figsize=(15,30))
plt.subplot(4, 1, 1)
sns.pointplot(china.index ,china.deaths, color = 'g')
plt.title("China's Deaths Cases Over Time" , fontsize = 25)
plt.xlabel('No. of Days', fontsize = 15)
plt.ylabel('Total cases', fontsize = 15)

plt.subplot(4, 1, 2)
sns.pointplot(us.index ,us.deaths, color = 'g')
plt.title("US's Deaths Cases Over Time" , fontsize = 25)
plt.xlabel('No. of Days', fontsize = 15)
plt.ylabel('Total cases', fontsize = 15)

plt.subplot(4, 1, 3)
sns.pointplot(italy.index ,italy.deaths, color = 'g')
plt.title("Italy's Deaths Cases Over Time" , fontsize = 25)
plt.xlabel('No. of Days', fontsize = 15)
plt.ylabel('Total cases', fontsize = 15)


plt.subplot(4, 1, 4)
sns.pointplot(india.index ,india.deaths, color = 'g')
plt.title("India's Deaths Cases Over Time" , fontsize = 25)
plt.xlabel('No. of Days', fontsize = 15)
plt.ylabel('Total cases', fontsize = 15)

plt.subplots_adjust(bottom=0.01, top=0.9)


# In[42]:


plt.figure(figsize=(15,30))
plt.subplot(4, 1, 1)
sns.pointplot(china.index ,china.recovered, color = 'orange')
plt.title("China's Recovered Cases Over Time" , fontsize = 25)
plt.xlabel('No. of Days', fontsize = 15)
plt.ylabel('Total cases', fontsize = 15)

plt.subplot(4, 1, 2)
sns.pointplot(us.index ,us.recovered, color = 'orange')
plt.title("US's Recovered Cases Over Time" , fontsize = 25)
plt.xlabel('No. of Days', fontsize = 15)
plt.ylabel('Total cases', fontsize = 15)

plt.subplot(4, 1, 3)
sns.pointplot(italy.index ,italy.recovered, color = 'orange')
plt.title("Italy's Recovered Cases Over Time" , fontsize = 25)
plt.xlabel('No. of Days', fontsize = 15)
plt.ylabel('Total cases', fontsize = 15)

plt.subplot(4, 1, 4)
sns.pointplot(india.index ,india.recovered, color = 'orange')
plt.title("India's Recovered Cases Over Time" , fontsize = 25)
plt.xlabel('No. of Days', fontsize = 15)
plt.ylabel('Total cases', fontsize = 15)

plt.subplots_adjust(bottom=0.01, top=0.9)


# In[46]:


pip install streamlit


# In[99]:


import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.title('Covid-19 World Cases')
st.write("It shows **Coronavirus Cases** Worldwide")
st.sidebar.title("Selector")
image = Image.open("coronavirus.png")
st.image(image, use_column_width=True)
st.markdown('<style>body{background-color: lightblue;}</style>', unsafe_allow_html=True)

@st.cache
def load_data():
    df = pd.read_csv("worldometer_data.csv")
    return df
df = load_data()

visualization = st.sidebar.selectbox('Select a Chart type', ('Bar Chart', 'Pie Chart', 'Line Chart'))
country_select = st.sidebar.selectbox('Select a country', df['Country/Region'].unique())
status_select = st.sidebar.radio('Covid-19 patient status', ('TotalCases', 'TotalDeaths', 'TotalRecovered', 'ActiveCases'))
selected_country = df[df['Country/Region']==country_select]
st.markdown("##*World level analysis*")

def get_total_dataframe(df):
    total_dataframe = pd.DataFrame({'Status': ['Confirmed', 'Recovered', 'Deaths', 'Active'],
                                    'Number of cases': [df.iloc[0]['TotalCases'],
                                                       df.iloc[0]['TotalRecovered'],
                                                       df.iloc[0]['TotalDeaths'],
                                                       df.iloc[0]['ActiveCases']]})
    return total_dataframe
country_total = get_total_dataframe(selected_country)

if visualization=='Bar Chart':
    country_total_graph = px.bar(country_total, x='Status', y='Number of cases', labels={'Number of cases'},
                                 title='{} : Number of cases in %'.format(country_select), color='Status')
    st.plotly_chart(country_total_graph)
elif visualization=='Pie Chart':
    if status_select=='TotalCases':
        st.title("Total Confirmed Cases")
        fig = px.pie(df, values=df[df['Country/Region']==country_select]['TotalCases'],
                     names=df[df['Country/Region']==country_select]['Country/Region'])
        st.plotly_chart(fig)
    elif status_select=='ActiveCases':
        st.title("Total Active Cases")
        fig = px.pie(df, values=df[df['Country/Region']==country_select]['ActiveCases'],
                     names=df[df['Country/Region']==country_select]['Country/Region'])
        st.plotly_chart(fig)
    elif status_select=='TotalDeaths':
        st.title("Total Death Cases")
        fig = px.pie(df, values=df[df['Country/Region']==country_select]['TotalDeaths'],
                     names=df[df['Country/Region']==country_select]['Country/Region'])
        st.plotly_chart(fig)
    else:
        st.title("Total Recovered Cases")
        fig = px.pie(df, values=df[df['Country/Region']==country_select]['TotalRecovered'],
                     names=df[df['Country/Region']==country_select]['Country/Region'])
        st.plotly_chart(fig)
elif visualization=='Line Chart':
    if status_select=='TotalDeaths':
        st.title("Total Death Cases Among Countries")
        fig = px.line(df, x='Country/Region', y='TotalDeaths', title='Total Death Cases Across Countries')
        st.plotly_chart(fig)


# In[100]:


import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.title('Covid-19 World Cases')
st.write("It shows **Coronavirus Cases** Worldwide")
st.sidebar.title("Selector")
image = Image.open("coronavirus.png")
st.image(image, use_column_width=True)
st.markdown('<style>body{background-color: lightblue;}</style>', unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv("worldometer_data.csv")
    return df

df = load_data()

visualization = st.sidebar.selectbox('Select a Chart type', ('Bar Chart', 'Pie Chart', 'Line Chart'))
country_select = st.sidebar.selectbox('Select a country', df['Country/Region'].unique())
status_select = st.sidebar.radio('Covid-19 patient status', ('TotalCases', 'TotalDeaths', 'TotalRecovered', 'ActiveCases'))
selected_country = df[df['Country/Region']==country_select]
st.markdown("##*World level analysis*")

def get_total_dataframe(df):
    total_dataframe = pd.DataFrame({'Status': ['Confirmed', 'Recovered', 'Deaths', 'Active'],
                                    'Number of cases': [df.iloc[0]['TotalCases'],
                                                       df.iloc[0]['TotalRecovered'],
                                                       df.iloc[0]['TotalDeaths'],
                                                       df.iloc[0]['ActiveCases']]})
    return total_dataframe

country_total = get_total_dataframe(selected_country)

if visualization=='Bar Chart':
    country_total_graph = px.bar(country_total, x='Status', y='Number of cases', labels={'Number of cases'},
                                 title='{} : Number of cases in %'.format(country_select), color='Status')
    st.plotly_chart(country_total_graph)
elif visualization=='Pie Chart':
    if status_select=='TotalCases':
        st.title("Total Confirmed Cases")
        fig = px.pie(df, values=df[df['Country/Region']==country_select]['TotalCases'],
                     names=df[df['Country/Region']==country_select]['Country/Region'])
        st.plotly_chart(fig)
    elif status_select=='ActiveCases':
        st.title("Total Active Cases")
        fig = px.pie(df, values=df[df['Country/Region']==country_select]['ActiveCases'],
                     names=df[df['Country/Region']==country_select]['Country/Region'])
        st.plotly_chart(fig)
    elif status_select=='TotalDeaths':
        st.title("Total Death Cases")
        fig = px.pie(df, values=df[df['Country/Region']==country_select]['TotalDeaths'],
                     names=df[df['Country/Region']==country_select]['Country/Region'])
        st.plotly_chart(fig)
    else:
        st.title("Total Recovered Cases")
        fig = px.pie(df, values=df[df['Country/Region']==country_select]['TotalRecovered'],
                     names=df[df['Country/Region']==country_select]['Country/Region'])
        st.plotly_chart(fig)


# In[ ]:





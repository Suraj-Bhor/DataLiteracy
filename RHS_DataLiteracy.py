#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import json
import plotly.io as pio
import plotly.offline as pl
import plotly.graph_objs as go
import plotly.express as px
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
import shapefile as shp

pl.init_notebook_mode()
pio.renderers.default = "browser"
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# In[2]:



dirname = '/Users/surajbhor/Documents/RHS/RHS_Data_Literacy/'
xls = 'RHS_Combined_2009.xlsx'
df = pd.read_excel(open(dirname+xls, 'rb'),index_col=None,header=0,sheet_name='2009')
df = df.drop(['Unnamed: 0'],axis=1)


# In[3]:


final_columns = ['ST','F_SC','F_CHC','F_PHC','F_CHC,DOC_PHC_P', 'SUR_CHC_P', 'OBGY_CHC_P','GDMO_AL_CHC_P', 'RAD_CHC_P','PHARMA_PHC_CHC_P','LT_PHC_CHC_P','NURSE_PHC_CHC_P',
'PHC_W_LR_P', 'PHC_W_OT_P','PHC_W_4BED_P','PHC_WO_ES_P','PHC_WO_RWS_P','PHC_WO_MAR_P','PHC_W_RT','CHC_W_FL','CHC_W_FOT','CHC_W_FLR','CHC_W_30BED',
'CHC_W_FXRAY','CHC_W_RT','AVG_RPC_PHC','AVG_RPC_CHC','N_FRU','FRU_PHC','FRU_CHC','FRU_W_30BED_P','FRU_W_OT_P','FRU_W_LR_P','FRU_W_BS_P','DOC_PHC_T_P','SUR_CHC_T_P','OBGY_CHC_T_P',
'RAD_CHC_T_P','PHARMA_PHC_CHC_T_P','LT_PHC_CHC_T_P','NURSE_PHC_CHC_T_P','AVG_RA_PHC','AVG_RA_CHC', 'AVG_RD_PHC','AVG_RD_CHC']


# In[4]:


df = df.drop(columns=[col for col in df if col not in final_columns])


# In[5]:


directory = os.fsencode(dirname)
filelist = []
dataframe_list = []
df_all_1 = pd.DataFrame()
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".xlsx"): 
         filelist.append(filename)  
         year = re.search("[0-9]{4}", filename)
         print(filename)
         df = pd.read_excel(open(dirname+filename, 'rb'),index_col=None,header=0,sheet_name=year.group(0))
         df = df.drop(columns=[col for col in df if col not in final_columns])
         df = df.assign(Year = year.group(0))
         df_all_1 = df_all_1.append(df)


# In[6]:


remove_str_1 = "^ Sanctioned data for 2013 used"
remove_str_2 = "@ Data for 2010 repeated"
remove_str_3 = "++ Sanctioned data for 2011 used"
# remove_str_3 = ""
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df_all_1 = df_all_1[cols]
df_all_1 = df_all_1.sort_values(by=['Year' , 'ST'])
df_all_1 = df_all_1.reset_index(drop=True)
df_all_1.replace('NaN', np.nan, inplace=True)
df_all_1.replace('-', np.nan, inplace=True)
df_all_1.replace(remove_str_1, np.nan, inplace=True)
df_all_1.replace(remove_str_2, np.nan, inplace=True)
df_all_1.replace(remove_str_3, np.nan, inplace=True)
df_all_1 = df_all_1.dropna(axis=1, how='all')
df_all_1 = df_all_1.replace(',','', regex=True)
df_all_1 = df_all_1.replace("\*",'', regex=True)


# In[7]:


df_all_1 = df_all_1.replace(to_replace ="A& N Islands",
                 value ="Andaman & Nicobar Island")
df_all_1 = df_all_1.replace(to_replace ="D & N Haveli",
                 value ="Dadara & Nagar Havelli")
df_all_1 = df_all_1.replace(to_replace ="Delhi",
                 value ="NCT of Delhi")
df_all_1 = df_all_1.replace(to_replace ="Orissa",
                 value ="Odisha")
df_all_1 = df_all_1.replace(to_replace ="Arunachal Pradesh",
                 value ="Arunanchal Pradesh")
df_all_1 = df_all_1.replace(to_replace ="Arunachal Pradesh",
                 value ="Arunanchal Pradesh")


# In[8]:


df_all_1.isnull().sum()
df_all_1 = df_all_1.fillna(0)
df_all_1


# In[33]:


func_HC = df_all_1[df_all_1['ST'].str.match('All India')][['Year','F_PHC','F_SC','F_CHC']]
func_personnel = df_all_1[df_all_1['ST'].str.match('All India')][['Year','SUR_CHC_P','OBGY_CHC_P','RAD_CHC_P']]


# In[31]:


fig, axs = plt.subplots(1, 3,figsize=(20,5))
axs[0].set_title('No. of Primary Healthcare Centers from 2009 to 2018')
axs[0].set_xlabel('Year')
axs[0].set_ylabel('Number of PHCs (All India)')
axs[0].plot(func_HC['Year'],func_HC['F_PHC'])

axs[1].set_title('No. of Sub Healthcare Centers from 2009 to 2018')
axs[1].set_xlabel('Year')
axs[1].set_ylabel('Number of SCs (All India)')
axs[1].plot(func_HC['Year'],func_HC['F_SC'])

axs[2].set_title('No. of Community Healthcare Centers from 2009 to 2018')
axs[2].set_xlabel('Year')
axs[2].set_ylabel('Number of CHCs (All India)')
axs[2].plot(func_HC['Year'],func_HC['F_CHC'])
fig.tight_layout()


# In[34]:


fig, axs = plt.subplots(1, 3,figsize=(20,5))
axs[0].set_title('No. of Surgeons present in CHCs')
axs[0].set_xlabel('Year')
axs[0].set_ylabel('Number of Surgeons (All India)')
axs[0].plot(func_personnel['Year'],func_personnel['SUR_CHC_P'])

axs[1].set_title('No. of Obstetrician-Gynecologist in CHCs')
axs[1].set_xlabel('Year')
axs[1].set_ylabel('Number of OBGYN (All India)')
axs[1].plot(func_personnel['Year'],func_personnel['OBGY_CHC_P'])

axs[2].set_title('No. of Radiographers in CHCs')
axs[2].set_xlabel('Year')
axs[2].set_ylabel('Number of Radiographers (All India)')
axs[2].plot(func_personnel['Year'],func_personnel['RAD_CHC_P'])
fig.tight_layout()


# In[11]:


from sklearn.preprocessing import StandardScaler
X_train_data = df_all_1.drop(columns=['ST','Year'])
x = X_train_data
x = StandardScaler().fit_transform(x) # normalizing the features
x


# In[12]:


np.mean(x), np.std(x)


# In[13]:


from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
X_train_data = df_all_1.drop(columns=['ST','Year'])
X_pca = pca.fit_transform(X_train_data)


# In[14]:


X_pca_df = pd.DataFrame(data = X_pca
             , columns = ['principal component 1', 'principal component 2'])


# In[15]:


X_pca_df


# In[16]:


exp_var_pca = pca.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(exp_var_pca)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())


# In[17]:


# Create the visualization plot
#
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# In[18]:


import seaborn as sns
plt.figure(figsize=(10,5))
sns.scatterplot(
    x="principal component 1", y="principal component 2",
    palette=sns.color_palette("rocket_r", 10),
    data=X_pca_df,
    legend="full",
    alpha=0.3
)


# In[19]:


import plotly.io as pio
import plotly.offline as pl
import plotly.graph_objs as go
import plotly.express as px
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
import shapefile as shp

pl.init_notebook_mode()
pio.renderers.default = "svg"


# In[20]:


df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",
                   dtype={"fips": str})
df.head()


# In[21]:



indian_states = json.load(open("/Users/surajbhor/Downloads/states_india.geojson","r"))


# In[22]:


state_id_map = {}
for feature in indian_states["features"]:
    feature["id"] = feature["properties"]["state_code"]
    state_id_map[feature["properties"]["st_nm"]] = feature["id"]


# In[23]:


df_india = df_all_1[df_all_1.loc[:,'ST'] != "All India"]
df_india.loc[:,'ID'] = df_india.loc[:,'ST'] .apply(lambda x: state_id_map[x])


# In[24]:


df_india


# In[25]:


fig = px.choropleth(df_india, geojson=indian_states, 
                    locations='ID',
                    color='F_SC',
                    hover_name= 'ST',
                    hover_data=['F_SC'],
                    title="Colormap of India showing number of functional Sub Centers for all Indian states.",
                    )

fig.update_geos(fitbounds="locations", visible=False)
fig.update_traces()
fig.show()


# In[26]:


fig = px.choropleth(df_india, geojson=indian_states, 
                    locations='ID',
                    color='F_PHC',
                    hover_name= 'ST',
                    hover_data=['F_PHC'],
                    title="Colormap of India showing number of functional PHCs for all Indian states.",
                    )

fig.update_geos(fitbounds="locations", visible=False)
fig.update_traces()
fig.show()


# In[27]:


fig = px.choropleth(df_india, geojson=indian_states, 
                    locations='ID',
                    color='F_CHC',
                    hover_name= 'ST',
                    hover_data=['F_CHC'],
                    title="Colormap of India showing number of functional CHCs for all Indian states.",
                    )

fig.update_geos(fitbounds="locations", visible=False)
fig.update_traces()
fig.show()


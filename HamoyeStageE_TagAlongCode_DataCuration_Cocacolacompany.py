#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


url_1= 'https://investors.coca-colacompany.com/filings-reports/all-sec-filings/content/0000021344-12-000007/a2011123110-k.htm#sFD60D13C2AB8AE15B2B55FE75624D861'
url_2 = 'https://investors.coca-colacompany.com/filings-reports/all-sec-filings/content/0000021344-15-000005/a2014123110-k.htm#sADED6865E781876F124F634A1CDE019D'
url_3 = 'https://investors.coca-colacompany.com/filings-reports/all-sec-filings/content/0000021344-18-000008/a2017123110-k.htm'
url_4 = 'https://investors.coca-colacompany.com/filings-reports/all-sec-filings/content/0000021344-20-000006/a2019123110-k.htm'


# In[3]:


dataset_1 = pd.read_html(url_1)
dataset_2 = pd.read_html(url_2)
dataset_3 = pd.read_html(url_3)
dataset_4 = pd.read_html(url_4)


# In[4]:


dataset_1[0].head()


# In[5]:


dataset_2[0].head()


# In[6]:


dataset_3[0].head()


# In[7]:


dataset_4[0].head()


# In[8]:


x = [0, 2, 6, 10]
y = [0, 2, 5, 8]

dataset_1 = dataset_1[90][x]
dataset_2 = dataset_2[88][x]
dataset_3 = dataset_3[90][x]
dataset_4 = dataset_4[122][y]


# In[9]:


dataset_1.head(10)


# In[10]:


dataset_2.head(10)


# In[11]:


dataset_3.head(10)


# In[12]:


dataset_4.head(10)


# In[13]:


pd.options.mode.chained_assignment = None


# In[14]:


for dataset in [dataset_1, dataset_2, dataset_3, dataset_4]:
    dataset.dropna(axis=0, how='all', inplace=True)


# In[15]:


dataset_1.head(5)


# In[16]:


dataset_2.head(5)


# In[17]:


dataset_3.head(5)


# In[18]:


dataset_4.head(5)


# In[19]:


for i in [dataset_1, dataset_2, dataset_3]:
    i.drop(index=[3,4], inplace=True)
    
dataset_4.drop(index=2, columns=8, inplace=True)


# In[20]:


dataset_1.columns = ['Year Ended December 31', '2011', '2010', '2009']
dataset_2.columns = ['Year Ended December 31', '2014', '2013', '2012']
dataset_3.columns = ['Year Ended December 31', '2017', '2016', '2015']
dataset_4.columns = ['Year Ended December 31', '2019', '2018']


# In[21]:


for n in [dataset_1, dataset_2, dataset_3, dataset_4]:
    n.reset_index(drop=True, inplace=True)


# In[22]:


dataset_1.head(0)


# In[23]:


dataset_1.head(5)


# In[24]:


dataset_2.head(5)


# In[25]:


dataset_3.head(5)


# In[26]:


dataset_4.head(5)


# In[27]:


for a in [dataset_1, dataset_2, dataset_3, dataset_4]:
    a['Year Ended December 31'] = a['Year Ended December 31'].str.upper()


# In[28]:


dataset_1.head(5)


# In[29]:


dataset_3.drop(index=[15,16], inplace=True)
dataset_3.reset_index(drop=True, inplace=True)


# In[30]:


new_values = ['INCOME BEFORE INCOME TAXES', 'INCOME TAXES']
index_values = [12, 13]

for o, index in enumerate(index_values):
    dataset_3.at[index, 'Year Ended December 31'] = new_values[o]


# In[31]:


new_row = pd.DataFrame(index=[2.5], columns=dataset_4.columns)
dataset_4 = pd.concat([dataset_4.iloc[:2], new_row, dataset_4.iloc[2:]]).reset_index(drop=True)
dataset_4.iloc[2]


# In[32]:


new_values = ['GROSS PROFIT', 22647.00, 21233]
i = 0
while i < 3:  
    for column in dataset_4.columns:
        dataset_4.at[2, column] = new_values[i]
        i += 1
    
dataset_4.iloc[2]


# In[33]:


dataset_4.at[10, '2018'] = -1674.00


# In[34]:


dataset_4['2018'] = dataset_4['2018'].astype(float)


# In[35]:


dataset_4.at[3, '2019'] = (dataset_4.at[2, '2019']/dataset_4.at[0, '2019'])*100
dataset_4.at[3, '2018'] = (dataset_4.at[2, '2018']/dataset_4.at[0, '2018'])*100

dataset_4.at[3, 'Year Ended December 31'] = 'GROSS PROFIT MARGIN'
dataset_4.iloc[3]


# In[36]:


dataset_4.drop(index=[18,19,20], inplace=True)


# In[37]:


new_row = pd.DataFrame(index=[7], columns=dataset_4.columns)
dataset_4 = pd.concat([dataset_4.iloc[:7], new_row, dataset_4.iloc[7:]]).reset_index(drop=True)


# In[38]:


new_values = ['OPERATING MARGIN', (dataset_4.at[6, '2019']/dataset_4.at[0, '2019']*100), (dataset_4.at[6, '2018']/dataset_4.at[0, '2018']*100)]
i = 0
while i < 3:  
    for column in dataset_4.columns:
        dataset_4.at[7, column] = new_values[i]
        i += 1
    
dataset_4.iloc[7]


# In[39]:


new_row = pd.DataFrame(index=[14], columns=dataset_4.columns)
dataset_4 = pd.concat([dataset_4.iloc[:14], new_row, dataset_4.iloc[14:]]).reset_index(drop=True)


# In[40]:


new_values = ['EFFECTIVE TAX RATE', (dataset_4.at[13, '2019']/dataset_4.at[12, '2019']*100), (dataset_4.at[13, '2018']/dataset_4.at[12, '2018']*100)]
i = 0
while i < 3:  
    for column in dataset_4.columns:
        dataset_4.at[14, column] = new_values[i]
        i += 1
    
dataset_4


# In[41]:


dataset_4.at[16, 'Year Ended December 31'] = 'LESS: NET INCOME ATTRIBUTABLE TO NONCONTROLLING INTERESTS'


# In[42]:


dataset_2


# In[43]:


dataset_2.at[11, '2014'] = -1263


# In[44]:


dataset_2


# In[45]:


dataset_3


# In[46]:


new_values = [-1666, 1234]
i = 0
while i < 2:  
    for column in ['2017', '2016']:
        dataset_3.at[11, column] = new_values[i]
        i += 1
    
dataset_3.iloc[11]


# In[49]:


for f in ['2017', '2016', '2015']:
    dataset_3[f] = dataset_3[f].astype(float)
    
dataset_3.dtypes


# In[50]:


dataset_2['2014'] = dataset_2['2014'].astype(float)


# In[51]:


dataset_3.at[17, 'Year Ended December 31'] = 'NET INCOME ATTRIBUTABLE TO SHAREOWNERS OF THE COCA-COLA COMPANY'


# In[52]:


merge_data = dataset_2.merge(dataset_1, on='Year Ended December 31', how='outer')
merge_data


# In[53]:


merge_two = dataset_4.merge(dataset_3, on='Year Ended December 31', how='outer')
merge_two


# In[54]:


merge_data.at[17, 'Year Ended December 31'] = 'NET INCOME ATTRIBUTABLE TO SHAREOWNERS OF THE COCA-COLA COMPANY'


# In[55]:


complete_dataset = merge_two.merge(merge_data, on='Year Ended December 31', how='outer')
complete_dataset['2019'] = round(complete_dataset['2019'], 2)
complete_dataset['2018'] = round(complete_dataset['2018'], 2)
complete_dataset


# In[56]:


complete_dataset.to_csv('COCACOLA(2019-2009).csv', index=False)


# In[57]:


graph = complete_dataset.iloc[[0]]
graph


# In[58]:


graph = graph.T
graph


# In[59]:


graph.reset_index(inplace=True)
graph


# In[60]:


graph.columns = ['Year', 'Revenue']
graph.drop(index = 0, inplace=True)
graph


# In[62]:


graph_two = complete_dataset.iloc[[0,2],:]
graph_two


# In[63]:


graph_two = graph_two.T
graph_two


# In[64]:


graph_two.reset_index(inplace=True)


# In[65]:


graph_two.columns = ['Year', 'Revenue', 'Profit']
graph_two.drop(index = 0, inplace=True)
graph_two


# In[76]:


graph


# In[79]:


graph = complete_dataset.iloc[[0]]
graph


# In[80]:


graph = graph.T
graph


# In[81]:


graph.reset_index(inplace=True)
graph


# In[82]:


graph.columns = ['Year', 'Revenue']
graph.drop(index = 0, inplace=True)
graph


# In[88]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 12))
ax.bar(graph['Year'], graph['Revenue'], color='grey')
ax.set_xlabel('Years', fontsize=14)
ax.set_ylabel('Revenue', fontsize=14)
ax.set_title('COCA-COLA SALES OVER A 10-YEAR PERIOD', fontsize=14)

for x, y in zip(graph['Year'], graph['Revenue']):
    ax.text(x, y, f"{y:,.0f}", ha='center', va='bottom', fontsize=11)

ax.tick_params(axis='both', labelsize=12)

ax.set_facecolor('#F5F1DD')

plt.show()


# In[89]:


graph_two = complete_dataset.iloc[[0,2],:]
graph_two


# In[90]:


graph_two = graph_two.T
graph_two


# In[91]:


graph_two.reset_index(inplace=True)


# In[92]:


graph_two.columns = ['Year', 'Revenue', 'Profit']
graph_two.drop(index = 0, inplace=True)
graph_two


# In[95]:


fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor('#F7EEB4')
ax.set_facecolor('#F7EEB4')

ax.plot(graph_two['Year'], graph_two['Revenue'], color='black', label='Revenue')
ax.plot(graph_two['Year'], graph_two['Profit'], color='red', label='Profit')

ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Amount ($)', fontsize=12)
ax.tick_params(axis='both', labelsize=12)

ax.set_title('Revenue and Profit over Time', fontsize=15)
ax.legend(loc='best')

plt.show()


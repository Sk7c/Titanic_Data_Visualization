#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np


# In[3]:


df = pd.read_csv('D:\\UDEMY\\Fundamental_Data_Analysis_Viz\\Titanic_Detailed.csv', index_col=0, encoding='unicode_escape')
df.head()


# In[4]:


#Histogram of Fare attribute using Matplotlib
plt.hist(df.Fare, color='red')


# In[11]:


#Distribution plot of Fare attribute using seaborn
sns.distplot(df.Fare, color='red')


# In[34]:


#Compare Fare based on age group
a = sns.jointplot(df.Fare, df.Age,kind='kde', color='blue')


# In[42]:


#Distibution of age group based on passenger class
sns.boxplot(x=df['Pclass'], y=df['Age'], palette='Blues')
plt.show()


# In[44]:


#Age concentration onboard
sns.violinplot(x=df['Embarked'], y=df['Age'], color='Green')


# In[76]:


#Number of passengers in each passenger class
sns.countplot(x='Pclass', data=df)


# In[5]:


import plotly.express as px
import pandas as pd
import numpy as np


# In[15]:


#heatmap to show correlation among attributes
df_copy = df.drop(['Name', 'SibSp', 'Parch', 'Cabin','Ticket','Age','Embarked'], axis =1)
corr = df_copy.corr()
labels = list(df_copy)
fig = px.imshow(df_copy,x=labels)
fig.show()
                 


# In[73]:


df.count()


# In[83]:



fig = px.violin(df, x='Pclass', y='Age')
fig.show()


# In[16]:


#density heatmap to show surivival rate based on passenger class


fig = px.density_heatmap(df, x='Pclass', y='Survived')
fig.show()


# In[ ]:





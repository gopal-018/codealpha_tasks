#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[11]:


df = pd.read_csv("heart.csv")
print(df)


# In[12]:


x=df.drop(columns=['target'])
y=df['target']

print(x)
print(y)


# In[13]:


df.fillna(df.median(), inplace=True)  # Replace missing values with the median


# In[14]:


# Detect and handle outliers using the IQR method
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Replace outliers with upper/lower bounds
df = df.clip(lower=lower_bound, upper=upper_bound, axis=1)
df


# In[17]:



scaler = StandardScaler()  # Use MinMaxScaler() for normalization
x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
x_scaled


# In[26]:


x_train, x_test, y_train, y_test = train_test_split(x_scaled,y, test_size=0.2, random_state=42)

x_train


# In[27]:


x_test


# In[28]:


y_train


# In[29]:


y_test


# In[39]:


#using normalization
from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
x_scale=pd.DataFrame(scale.fit_transform(x),columns=x.columns)
x_scale


# In[40]:


x_train, x_test, y_train, y_test = train_test_split(x_scale,y, test_size=0.2, random_state=42)

x_train


# In[42]:


x_train


# In[43]:


x_test


# In[44]:


y_train


# In[45]:


y_test


# In[ ]:





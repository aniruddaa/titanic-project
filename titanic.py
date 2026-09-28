#!/usr/bin/env python
# coding: utf-8

# In[67]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib 


# In[68]:


titanic=pd.read_csv("titanic.csv")


# In[69]:


titanic


# In[70]:


titanic.info()


# In[71]:


titanic.isnull().sum()


# In[72]:


titanic["Age"]=titanic["Age"].fillna(titanic["Age"].mean())


# In[73]:


titanic


# In[74]:


titanic["Cabin"]=titanic["Cabin"].fillna(titanic["Cabin"].mode()[0])


# In[75]:


titanic


# In[76]:


titanic.drop(columns=["Name"],inplace=True)


# In[77]:


titanic


# In[78]:


titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch'] + 1 

titanic['IsAlone'] = (titanic['FamilySize'] == 1).astype(int)


# In[79]:


titanic


# In[80]:


titanic = pd.get_dummies( 
titanic, 
columns=['Sex', 'Embarked'], 
drop_first=True)


# In[81]:


titanic


# In[82]:


x=titanic[["Age"]]


# In[83]:


y=titanic[["Survived"]]


# In[84]:


model=LogisticRegression()


# In[85]:


model.fit(x,y)


# In[86]:


model.predict([[90]])


# In[89]:


joblib.dump(model,"titanicmodel.pkl")


# In[ ]:





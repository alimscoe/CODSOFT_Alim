#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd


# In[120]:


import numpy as np
import warnings
warnings.filterwarnings('ignore') 


# In[121]:


df=pd.read_csv("C:/Users/ALIM/Downloads/archive (6)/tested.csv")


# In[122]:


df


# In[123]:


df.head()


# In[124]:


df.tail()


# In[125]:


df.size


# In[126]:


df.dtypes


# In[127]:


df.shape


# In[128]:


df.describe()


# In[129]:


df.count()


# In[130]:


df.isnull()


# In[131]:


df.isnull().sum()


# In[132]:


df["Age"] = df["Age"].fillna(df["Age"].mean())


# In[133]:


df["Fare"] = df["Fare"].fillna(df["Fare"].mean())


# In[134]:


df.isnull().sum()


# In[135]:


Embarkeds = df.Embarked.unique()


# In[136]:


for Embarked in Embarkeds:
    print("Embarked:", Embarked)


# In[137]:


df['Embarked'] = df["Embarked"].map({'Q':0, 'S':1, 'C':2}).astype(int)


# In[138]:


df['Embarked'].head()


# In[139]:


df["Sex"] = df["Sex"].map({'female':1, 'male':0}).astype(int)
df["Sex"].head(5)


# In[140]:


df.dtypes


# In[141]:


df["Age"] = df.Age.astype(int)


# In[142]:


df.dtypes


# In[143]:


df["Fare"] = df.Fare.astype(int)


# In[144]:


df.dtypes


# In[145]:


df1= df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace = True)


# In[146]:


df.head(5)


# In[147]:


df[(df["Sex"] == 1) & (df["Survived"]==1)]


# In[148]:


import seaborn as sb
import matplotlib.pyplot as mpl


# In[149]:


_, ax = mpl.subplots(figsize = (10, 7))
sb.histplot(data = df, x = "Age", hue = "Survived", multiple = "stack", kde = True, palette=["red", "blue"], ax = ax)
ax.set_title("Age with Survival")
ax.set_xlabel("Age")
ax.set_ylabel("Number of passengers")
ax.legend(title = "Legends", labels = ["Survived", "Not survived"])
mpl.plot()


# In[150]:


embarked_counts = df['Embarked'].value_counts()


# In[151]:


mpl.figure(figsize=(8, 8))
mpl.pie(embarked_counts, labels=embarked_counts.index, autopct='%1.1f%%', colors=['yellow', 'green', 'orange'])
mpl.title("Distribution of Passengers by Embarked")
mpl.legend(["Q", "S", "C"])
mpl.show()


# In[152]:


_, ax = mpl.subplots(figsize = (10, 7))
sb.countplot(data = df, x = "Embarked", hue = "Survived", palette=["blue", "orange"], ax = ax)
ax.set_title("Countplot for Embarked with Survival")
ax.set_xlabel("Embarked")
ax.set_xticklabels(["Q", "S", "C"])
ax.set_ylabel("Number of passengers")
ax.legend(title = "Legends", labels = ["Not Survived", "Survived"])
mpl.plot()


# In[153]:


_, ax = mpl.subplots(figsize = (10, 7))
sb.histplot(data = df, x = "Fare", hue = "Survived", multiple = "stack", kde = True, palette = ["red", "green"], ax = ax)
ax.set_title("Histogram for Fare with Survival")
ax.set_xlabel("Fare")
ax.set_ylabel("Number of passengers")
ax.legend(title = "Legends", labels = ["Survived", "Not survived"])
mpl.plot()


# In[154]:


ax = sb.countplot(data = df, x="Survived", hue = "Survived", palette = ["red", "green"])
ax.set_xlabel("Survived")
ax.set_ylabel("Count")
ax.set_title("Survival Count")
ax.legend(title = "Legends", labels = ["Not Survived", "Survived"])
mpl.xticks([0,1],["Male", "Female"])
mpl.show()


# In[155]:


from sklearn.model_selection import train_test_split


# In[156]:


Input = df.drop(['Survived'], axis=1)
Output = df.iloc[:,1]
x_train, x_test, y_train, y_test = train_test_split(Input, Output, test_size = 0.2, random_state = 1)


# In[157]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as score
LR = LogisticRegression()
LR.fit(x_train, y_train)


# In[158]:


PredictedValue = LR.predict(x_test)
print("Accuracy = {:0.2f}%".format(score(PredictedValue, y_test)*100))


# In[ ]:





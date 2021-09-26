#!/usr/bin/env python
# coding: utf-8

# # This is Rain fall predication in Australia and EDA.

# ### Importing Modules

# In[1]:


# importing modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings(action="ignore")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report,roc_curve, roc_auc_score


# ### Importing Dataset

# In[3]:


df = pd.read_csv("weatherAUS.csv",nrows=10000)
df.head().T


# In[4]:


# Info of data
df.info()


# In[5]:


# shape of data:
print(f'Number of columns: { df.shape[0]} and Number of rows: {df.shape[1]}')


# In[6]:


# Checking for null values
df.isna().sum()


# In[7]:


# statistical info of dataset
df.describe().T


# In[8]:


df.describe()


# In[9]:


# Identifying Continuous and Categorical Columns
category=[]
contin = []

for i in df.columns:
    if df[i].dtype =="object":
        category.append(i)
        
    else:
        contin.append(i)

print("Categorical:",category)
print("Continuous:", contin)


# In[10]:


df.head()


# **Encoding RainToday and RainTomorrow Columns** using LabelEncoder

# In[11]:


df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})
df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})


# In[12]:


df["RainToday"].unique()


# In[13]:


df["RainTomorrow"].unique()


# In[14]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[15]:


df[["RainToday","RainTomorrow"]]


# Percentage of **Null values in dataset**

# In[16]:


(df.isnull().sum()/len(df))*100


# In[17]:


(df.isnull().sum()/len(df))


# In[18]:


df.head().T


# In[19]:


df.columns


# ### Handling Null values

# In[20]:


# filling the missing values for continuous variables with mean
df["MinTemp"]= df["MinTemp"].fillna(df["MinTemp"].mean())
df["MaxTemp"]= df["MaxTemp"].fillna(df["MaxTemp"].mean())
df["Evaporation"]= df["Evaporation"].fillna(df["Evaporation"].mean())
df["Sunshine"]= df["Sunshine"].fillna(df["Sunshine"].mean())
df["WindGustSpeed"]= df["WindGustSpeed"].fillna(df["WindGustSpeed"].mean())
df["Rainfall"]= df["Rainfall"].fillna(df["Rainfall"].mean())
df["WindSpeed9am"]= df["WindSpeed9am"].fillna(df["WindSpeed9am"].mean())
df["WindSpeed3pm"]= df["WindSpeed3pm"].fillna(df["WindSpeed3pm"].mean())
df["Humidity9am"]= df["Humidity9am"].fillna(df["Humidity9am"].mean())
df["Humidity3pm"]= df["Humidity3pm"].fillna(df["Humidity3pm"].mean())
df["Pressure9am"]= df["Pressure9am"].fillna(df["Pressure9am"].mean())
df["Pressure3pm"]= df["Pressure3pm"].fillna(df["Pressure3pm"].mean())
df["Cloud9am"]= df["Cloud9am"].fillna(df["Cloud9am"].mean())
df["Cloud3pm"]= df["Cloud3pm"].fillna(df["Cloud3pm"].mean())
df["Temp9am"]= df["Temp9am"].fillna(df["Temp9am"].mean())
df["Temp3pm"]= df["Temp3pm"].fillna(df["Temp3pm"].mean())


# In[21]:


#Filling the missing values for continuous variables with mode
df['RainToday']=df['RainToday'].fillna(df['RainToday'].mode()[0])
df['RainTomorrow']=df['RainTomorrow'].fillna(df['RainTomorrow'].mode()[0])
df['WindDir9am'] = df['WindDir9am'].fillna(df['WindDir9am'].mode()[0])
df['WindGustDir'] = df['WindGustDir'].fillna(df['WindGustDir'].mode()[0])
df['WindDir3pm'] = df['WindDir3pm'].fillna(df['WindDir3pm'].mode()[0])


# In[22]:


df.head()


# In[23]:


# again checking for null values
(df.isnull().sum()/len(df))*100


# ### **Countplot** for RainToday and Raintomorrow:

# In[24]:


fig, ax =plt.subplots(1,2)
plt.figure(figsize=(8,5))
sns.countplot(df["RainToday"],ax=ax[0])
sns.countplot(df["RainTomorrow"],ax = ax[1])


# ### Heatmap showing **Correlation** among attributes of data

# In[25]:


#heatmap
plt.figure(figsize=(18,12))
sns.heatmap(df.corr(), annot=True)
plt.xticks(rotation=90)
plt.show()


# **Inferences from Heatmap**:
# * MinTemp and Temp9am highly correlated.
# * MinTemp and Temp3pm highly correlated.
# * MaxTemp and Temp9am highly correlated.
# * MaxTemp and Temp3pm highly correlated.
# * Temp3pm and Temp9am highly correlated.
# * Humidity9am and Humidity3pm highly correlated.

# In[26]:


#encoding remaining columns
df["Location"] = le.fit_transform(df["Location"])
df["WindDir9am"]= le.fit_transform(df["WindDir9am"])
df["WindDir3pm"]= le.fit_transform(df["WindDir3pm"])
df["WindGustDir"] = le.fit_transform(df["WindGustDir"])


# In[27]:


df.head()


# In[28]:


# Dropping highly correlated columns
df=df.drop(['Temp3pm','Temp9am','Humidity9am',"Date"],axis=1)
df.columns


# In[29]:


df.head()


# In[30]:


x=df.drop(['RainTomorrow','Location','WindGustDir','WindGustSpeed','WindDir3pm','WindDir9am','WindSpeed3pm','Pressure3pm','Cloud3pm','Evaporation','RainToday','Pressure9am','WindSpeed9am'],axis=1)
y=df['RainTomorrow']
x.columns


# ### Splitting data into Training and Testing Set

# In[31]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# ### RandomforestClassifier

# In[32]:


model=RandomForestClassifier()
model.fit(X_train,y_train)


# In[33]:


# accuracy of RandomForest Model
y_predxgb = model.predict(X_test)
report = classification_report(y_test, y_predxgb)
print(report)
print("Accuracy of the RandomForest Model is:",accuracy_score(y_test,y_predxgb)*100,"%")
cm = confusion_matrix(y_test, y_predxgb)
sns.heatmap(cm, annot=True,cmap="YlGnBu")
plt.title("Confusion Matrix for RandomForest Model")
plt.show()


# In[34]:


import pickle
pickle_out = open("Finalmodel.pkl","wb")
pickle.dump(model, pickle_out)
pickle_out.close()


# In[35]:


df


# In[ ]:





# In[ ]:





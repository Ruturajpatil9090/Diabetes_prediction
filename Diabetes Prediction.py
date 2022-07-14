#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('diabetes.csv')


# In[106]:


df.head()


# In[4]:


df.shape


# In[6]:


df['Outcome'].value_counts()


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


X = df.drop(columns=('Outcome'),axis=1)
y= df['Outcome']


# In[10]:


X.shape


# In[11]:


y.shape


# In[12]:


from sklearn.preprocessing import StandardScaler


# In[16]:


scaler = StandardScaler()
scaler.fit(X)
standarized_scaler = scaler.transform(X)


# In[17]:


standarized_scaler


# In[18]:


X = standarized_scaler
y=df['Outcome']


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=2)


# In[21]:


X_train.shape , X_test.shape


# # Model Building

# In[49]:


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[58]:


models = {'Logistic Reg':LogisticRegression(),
          'Linear Reg':LogisticRegression(),
          'KNN': KNeighborsClassifier(n_neighbors=5,algorithm='auto',leaf_size=30,metric='minkowski',n_jobs=1,metric),
          'SVM': SVC(kernel='linear',random_state=15,C=1.0),
          'RF': RandomForestClassifier(n_estimators=300,max_depth = 9,min_samples_leaf = 5,random_state=2),
         'Ada': AdaBoostClassifier(base_estimator=LogisticRegression(),n_estimators=150)}


# In[59]:


for name, model in models.items():
    model.fit(X_train,y_train)
    print(name, model.score(X_test,y_test))


# # Final model Selection

# In[102]:


rf_model = RandomForestClassifier(n_estimators=300,max_depth = 9,min_samples_leaf = 5,random_state=2)
rf_model.fit(X_train,y_train)
rf_model.score(X_train,y_train), rf_model.score(X_test,y_test)


# In[101]:


# rfc = RandomForestClassifier()
# parameters = {
#     "n_estimators":[5,10,50,100,250,300],
#     "max_depth":[2,4,8,9,16,32,None]
    
# }


# In[97]:


# from sklearn.model_selection import GridSearchCV


# In[98]:


# cv = GridSearchCV(rfc,parameters,cv=5)
# cv.fit(X_train,y_train)


# In[99]:


# def display(results):
#     print(f'Best parameters are: {results.best_params_}')
#     print("\n")
#     mean_score = results.cv_results_['mean_test_score']
#     std_score = results.cv_results_['std_test_score']
#     params = results.cv_results_['params']
#     for mean,std,params in zip(mean_score,std_score,params):
#         print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')


# In[100]:


# display(cv)


# In[109]:


input_data = (0,100,88,60,110,46.8,0.962,31)

num_array = np.asarray(input_data)

num_array_reshape = num_array.reshape(1,-1)

std_data = scaler.transform(num_array_reshape)

prediction = rf_model.predict(std_data)
print(prediction)


# In[110]:


import pickle

pickle.dump(rf_model,open('diabetes.pkl','wb'))
diabetes_prediction =pickle.load(open('diabetes.pkl','rb'))


# In[ ]:





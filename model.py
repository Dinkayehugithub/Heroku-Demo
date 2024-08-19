#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
data=pd.read_csv(r"C:\Anemia_Level_Dataset\Remove_Missing\SMOTE data - Copy.csv",sep=',',index_col='id')
print(data.shape)
data.head()


# In[10]:


X=data.drop('level',axis=1)
y=data.level


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)


# In[12]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(criterion='entropy',max_features='sqrt',min_samples_split=2,n_estimators=1,random_state=10,max_depth=100, max_leaf_nodes=1000, n_jobs=-1)
model.fit(X_train,y_train)
y_predict = model.predict(X_test)


# In[13]:


from sklearn.metrics import mean_absolute_error, r2_score
print("MAE : ", mean_absolute_error(y_test,y_predict))
r2_score(y_test,y_predict)


# In[14]:


import sklearn as extjoblib
import joblib
import pickle


# In[15]:


joblib.dump(model,'C:/Users/Net/Desktop/Anemia_Lev_Final_Work/Artifact/anemia_level_prediction_model.ml')


# In[16]:


import pickle
# Lets dump our SVM model
pickle.dump(model, open('C:/Users/Net/Desktop/Anemia_Lev_Final_Work/Artifact/anemia_level_prediction_model.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# In[3]:


X=pd.read_csv(r"E:\AnalytixLAB Internship\Oorja Crime Project\X.csv")
X


# In[4]:


X.info()


# In[5]:


y=pd.read_csv(r"E:\AnalytixLAB Internship\Oorja Crime Project\target.csv")
y


# In[7]:


label_encoder= LabelEncoder()
y_enc=label_encoder.fit_transform(y)
y_enc


# In[8]:


X_cat= X.select_dtypes('object')
X_num= X.select_dtypes(['float64', 'int64'])


# In[9]:


X_cat.nunique()


# In[10]:


ohe= OneHotEncoder(drop='first', sparse_output=False)
ohen=ohe.fit_transform(X_cat)
ohe.get_feature_names_out()


# In[11]:


X_cat=pd.DataFrame(ohen,columns=ohe.get_feature_names_out())
X_cat


# In[13]:


X_new=pd.concat([X_num,X_cat],axis=1)
X_new.head()


# In[24]:


selector = SelectKBest(score_func=f_classif, k=12)
X_selected = selector.fit_transform(X_new, y_enc)
X_selected = pd.DataFrame(X_selected, columns=X_new.columns[selector.get_support()])
X_selected.head()


# In[25]:


from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_selected, y_enc)


# In[26]:


X_res.shape


# In[27]:


#Train-test split
X_train, X_val, y_train, y_val=train_test_split(X_res, y_res, test_size=0.3, random_state=37)


# In[28]:


model1 = XGBClassifier(n_estimators=100, max_depth=10)
model1= model1.fit(X_train,y_train)
y_enc_pred = model1.predict(X_val)
y_enc_pred


# In[29]:


y_predictlabels= label_encoder.inverse_transform(y_enc_pred)
print(y_predictlabels)


# In[30]:


accuracy_score(y_val,y_enc_pred)


# In[31]:


print(classification_report(y_val,y_enc_pred))


# In[33]:


import joblib

joblib.dump({'model': model1, 'encoder_X': ohe, 'encoder_Y': label_encoder, 'selector': selector}, 
            'E:\\AnalytixLAB Internship\\Oorja Crime Project\\model_bundle.pkl')


# In[ ]:





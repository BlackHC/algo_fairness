
# coding: utf-8

# In[4]:


import blackhc.notebook


# In[50]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[51]:


get_ipython().magic('matplotlib inline')


# In[7]:


data = pd.read_csv('input/german_processed.csv')


# In[99]:


data = pd.read_csv('input/german_repaired.csv')


# In[100]:


data.describe()


# In[101]:


one_hot_data = pd.get_dummies(data)


# In[102]:


one_hot_data.dtypes


# In[103]:


from sklearn import linear_model


# In[104]:


logistic = linear_model.LogisticRegression()


# In[105]:


age_X = one_hot_data[one_hot_data.columns.difference(['credit', 'sex_male', 'sex_female'])]


# In[106]:


age_Y = one_hot_data['sex_male']


# In[107]:


logistic.fit(age_X, age_Y)


# In[125]:


logistic.predict(age_X)


# In[120]:


logistic.score(age_X, age_Y)


# In[109]:


p1 = np.matmul(age_X, logistic.coef_.T) - logistic.intercept_


# In[119]:


logistic.intercept_


# In[110]:


plt.plot(p1, np.zeros_like(p1), 'x')


# In[111]:


p_rest = age_X - logistic.coef_ * p1 / np.linalg.norm(logistic.coef_)**2


# In[117]:


np.matmul(p_rest, logistic.coef_.T) # Sanity check


# In[113]:


from sklearn import decomposition


# In[114]:


pca = decomposition.PCA(1)


# In[115]:


p2 = pca.fit_transform(p_rest) 


# In[116]:


for mask, color in zip([age_Y==0, age_Y==1], ['r', 'b']):
    plt.scatter(p1[mask], p2[mask], c=color, alpha=0.5, marker='x')
plt.show()


# In[121]:


np.count_nonzero(age_Y==0)


# In[122]:


np.count_nonzero(age_Y==1)


# In[126]:


1-271/(271+729.0)


# In[97]:


for mask, color in zip([age_Y==0, age_Y==1], ['r', 'b']):
    plt.scatter(p1[mask], p2[mask], c=color, alpha=0.5, marker='x')
plt.show()


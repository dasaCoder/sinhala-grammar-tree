#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd


# In[63]:


df = pd.read_csv("sinhala_grammar.csv")
df.head()


# In[64]:


inputs = df.drop('is_correct',axis='columns')
target = df['is_correct']


# In[65]:


inputs
target


# In[66]:


inputs


# In[67]:


from sklearn.preprocessing import LabelEncoder


# In[68]:


subject = LabelEncoder()
tense = LabelEncoder()
person = LabelEncoder()
sex = LabelEncoder()
verb_root = LabelEncoder()


# In[69]:


inputs['subject_en'] = subject.fit_transform(inputs['subject'])
inputs['tense_en'] = tense.fit_transform(inputs['tense'])
inputs['person_en'] = person.fit_transform(inputs['person'])
inputs['sex_en'] = sex.fit_transform(inputs['sex'])
inputs['verb_root_en'] = verb_root.fit_transform(inputs['verb_root'])
inputs.head()


# In[70]:


encoded_inputs = inputs.drop(['subject','tense','sex','verb_root','person'],axis='columns')
encoded_inputs


# In[71]:


from sklearn import tree


# In[72]:


model = tree.DecisionTreeClassifier()


# In[73]:


model.fit(encoded_inputs,target)


# In[74]:


model.score(encoded_inputs,target)


# In[75]:


#model.predict([[1,1,0,0,0]])


# In[76]:


print(model.predict([[1,1,0,0,0]]))


# In[77]:


print(model.predict([[1,1,0,0,3]]))


# In[ ]:





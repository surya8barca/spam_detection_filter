#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import nltk


# In[ ]:


nltk.download_shell() #for downloading stopwords package


# In[ ]:


#getting the dataset of messages


# In[5]:


messages=pd.read_csv('smsspamcollection/SMSSpamCollection',sep='\t',names=['label','message'])


# In[6]:


print(len(messages))


# In[7]:


messages.head()


# In[12]:


messages.groupby('label').describe()


# In[13]:


messages['length']=messages['message'].apply(len)


# In[14]:


messages.head()


# In[20]:


messages['length'].plot.hist(bins=50,figsize=(10,5))


# In[23]:


messages.hist(column='length',by='label',bins=50,figsize=(12,5))


# In[24]:


#cleaning data


# In[25]:


import string


# In[54]:


from nltk.corpus import stopwords 


# In[67]:


def clean_text(str):
    
    #removing punctuation
    clean_pun=[ch for ch in str if ch not in string.punctuation]
    clean_pun1=''.join(clean_pun)
    
    #removing stopwords
    clean_stop1=list(clean_pun1.split( ))
    clean_str=[x for x in clean_stop1 if x not in stopwords.words('english')]
    
    
    return clean_str


# In[68]:


messages['message'].head()


# In[69]:


messages['message'].head().apply(clean_text)


# In[70]:


#vectorizaion


# In[72]:


from sklearn.feature_extraction.text import CountVectorizer


# In[108]:


bow=CountVectorizer(analyzer=clean_text).fit(messages['message'])


# In[110]:


len(bow.vocabulary_)


# In[114]:


messages_bow=bow.transform(messages['message'])


# In[115]:


messages_bow.shape #shape of sparse matrix


# In[122]:


messages_bow.nnz #non zero values in sparse matrix


# In[119]:


#finding TFIDF values


# In[130]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[131]:


tfidf=TfidfTransformer().fit(messages_bow)


# In[132]:


messages_tfidf=tfidf.transform(messages_bow)


# In[141]:


tfidf.idf_[bow.vocabulary_['word']] #tfidf value for words


# In[142]:


#model for prediction (MultinomialNB)


# In[143]:


from sklearn.naive_bayes import MultinomialNB


# In[144]:


model=MultinomialNB()


# In[145]:


model.fit(messages_tfidf,messages['label'])


# In[149]:


pred1=model.predict(messages_tfidf)


# In[152]:


pred1


# In[148]:


#now training the data and then testing the model


# In[156]:


from sklearn.model_selection import train_test_split


# In[157]:


msg_train,msg_test,label_train,label_test=train_test_split(messages['message'],messages['label'],test_size=0.4)


# In[158]:


#using pipeline for performing repeated operations  


# In[159]:


from sklearn.pipeline import Pipeline


# In[164]:


pipe=Pipeline([
    ('bow',CountVectorizer(analyzer=clean_text)),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB())
])


# In[165]:


pipe.fit(msg_train,label_train)


# In[166]:


prediction=pipe.predict(msg_test)


# In[167]:


prediction


# In[168]:


#evaluate the model


# In[169]:


from sklearn.metrics import classification_report,confusion_matrix


# In[172]:


print(confusion_matrix(label_test,prediction))
print()
print(classification_report(label_test,prediction))


# In[ ]:





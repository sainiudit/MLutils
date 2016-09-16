
# coding: utf-8

# In[203]:

import re
import string
import pandas as pd
import funcy as fp
import pyLDAvis
from gensim import corpora
pyLDAvis.enable_notebook()
#import regex
import re

#start process_tweet
def processTweet(tweet):
    # process the tweets

    #Convert to lower case
    tweet = re.sub('RT ',' ',tweet)
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+',' ',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end

#Read the tweets one by one and process it


# In[204]:

data=pd.read_csv("/home/udit/research/learn/talkingdata/tweets/tweets.csv")
data=data[~data['user_timeline'].str.contains("ScootsyIt").astype(bool)]
data=data[data.retweeted_status.isnull()]
data['text']=data['text'].apply(lambda x:x.decode('utf-8'))
data['text']=data.text.apply(processTweet)


# In[205]:

documents =list(data.text)


# In[206]:

from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
stoplist = 'for a of the and to in we are as you  can do it is us be'.split()
stoplist.append(stop)


# In[207]:

texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]


# In[195]:

from nltk.stem.porter import PorterStemmer
p_stemmer = PorterStemmer()


# In[196]:

texts=[[p_stemmer.stem(word) for word in doc] for doc in texts]


# In[208]:

from gensim import corpora, models

dictionary = corpora.Dictionary(texts)


# In[209]:

corpus = [dictionary.doc2bow(text) for text in texts]


# In[210]:

import gensim
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=20)


# In[217]:

import pyLDAvis.gensim
pyLDAvis.enable_notebook()
html=pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)


# In[223]:

pyLDAvis.save_html(html,"/home/udit/research/learn/talkingdata/tweets/ldavis.html")


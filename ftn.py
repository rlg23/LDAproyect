#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##Functions


# In[7]:


import config
import logging

from IPython import display
from dataset.TPL import *
import dataset.preprocessing as pre

import numpy as np
import pandas as pd
import time 
import gensim 
from gensim import corpora, models, similarities
from gensim.models import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
logging.basicConfig(level=logging.INFO)


# In[2]:


class NumbersAndTelescopes(pre.Numbers, pre.Telescopes):
    pass

colorize1 = pre.Numbers()
colorize2 = NumbersAndTelescopes()


# In[3]:

#For 2) Prepocessing...

def generate_serie(obs): 
    #we create the first row for our serie:
    firstrow=[]
    rows=[]
    for a in obs.load_trace(0)["event"]:
        firstrow.append(colorize2.color(a))
    #and then we create the serie:
    serie= pd.Series([firstrow])
    
    #Finally we repeat this process to create the entire serie
    # for i in range(1, len(obs.index)):
    for i in obs.index.index:
    
        for a in obs.load_trace(i)["event"]:
            rows.append(colorize2.color(a))
                
        serie=serie.append(pd.Series([rows],index=[i]))
        rows=[]
    
    return serie


def generate_serie_SES(obs): 
    chunks=[]
    for i in obs.index.index.values:
        T = obs.load_trace(i)
        T['color'] = T['event'].apply( lambda x: colorize2.color(x).replace(' ','_')  )
        chunks.append( pd.Series([ list(T['color'].values) ],index=[i]) )

    return pd.concat(chunks)

def generate_serie_SESmalaaaaa(obs): 
    #we create the first row for our serie:
    firstrow=[]
    rows=[]
    for a in obs.load_trace(0)["event"]:
        firstrow.append(colorize2.color(a))
        #borramos el espacio del comienzo y final
        firstrow=list(map(lambda x: x.strip(), firstrow))
        #hacemos el cambio para la primera fila:
        firstrow=list(map(lambda x: x.replace(' ','_'), firstrow))
        
    #and then we create the serie:
    serie= pd.Series([firstrow])
    
    #Finally we repeat this process to create the entire serie
    # for i in range(1, len(obs.index)):
    for i in obs.index.index:
        for a in obs.load_trace(i)["event"]:
            rows.append(colorize2.color(a))
            #borramos el espacio del comienzo y final de los demás
            rows=list(map(lambda x: x.strip(), rows))
            #Expandimos el cambio para el resto de la data
            rows=list(map(lambda x: x.replace(' ','_'), rows))
                
        serie=serie.append(pd.Series([rows],index=[i]))
        rows=[]
    
    return serie

import re
#Implementamos un tokenizador que separa las palabras unicamente cuando encuentre espacios.
def my_tokenizer(text):
    # split based on whitespace
    return re.split("\\s+",text)

#With gensim we create the dictionary and BoW easily
def DictionaryandBow(obs):
    #for dictionary:
    dictionary=corpora.Dictionary(obs)
    #for Bow:
    bow_corpus= [dictionary.doc2bow(doc) for doc in obs]
    
    return dictionary, bow_corpus

def vectorizer(info_train):
    
    #Info_train: data prepocessed for training 
    
    #Clase del CountVectorizer() con el tokenizador implementado
    vect=CountVectorizer(tokenizer=my_tokenizer,lowercase=False)
    
    #Dejamos todo como str (la info contiene números)
    info_train=info_train.apply(str)
    
    #Aplicamos el C_V y obtenemos la matrix token-frequency
    info_train = vect.fit_transform(info_train)
    
    #return de matrix token-frequency in info_train and the class CountVectorizer()
    return info_train, vect

#3) For model LDA:

#return an integer list of predicted topic catergories for a given topic matrix
def get_keys(topic_matrix):
    #topic_matrix is the top matrix created with  ldamodel.fit_transform(info_train_v)
    
    # print(topic_matrix.argmax(axis = 1)) # axis = 1, will return maximum index in that array 
    keys = topic_matrix.argmax(axis = 1).tolist()
    print("length of the keys is: ",len(keys))
    return keys

#Return a tuple of topic categories and their accompanying magnitude for a given list of keys
from collections import Counter
def key_to_count(keys):
    count_pairs = Counter(keys).items()
    # print("Count_pairs",count_pairs)
    categories = [pair[0] for pair in count_pairs]
    # print("categories",categories)
    counts = [pair[1] for pair in count_pairs]
    # print("Counts: ",counts)
    return (categories, counts)

#4) For Vis:

def topic_dom(lda_model,info_vec):
    #lda_model: modelo LDA ya creado
    #info_vec: matriz token-frequency
    
    lda_output = lda_model.transform(info_vec)
    # column names
    topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]
    # index names
    docnames = ["Doc" + str(i) for i in range(info_vec.shape[0])]
    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)
    
    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic
    
    return df_document_topic['dominant_topic']

def topic_document(lda_model, info_vec):
    #lda_model: modelo LDA ya creado
    #info_vec: matriz token-frequency
    
    doc_topic = lda_model.transform(info_vec)
    doc_topic_df = pd.DataFrame(data=doc_topic)
    return doc_topic_df

def topic_distribution(lda_model, info_vec):
    
    #lda_model: modelo LDA ya creado
    #info_vec: matriz token-frequency
    
    lda_output = lda_model.transform(info_vec)
    # column names
    topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]
    # index names
    docnames = ["Doc" + str(i) for i in range(info_vec.shape[0])]
    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)
    
    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic
    

    df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
    df_topic_distribution.columns = ['Topic Num', 'Num Documents']
    
    return  df_topic_distribution

def topic_keyword(lda_model, vect_class):
    #lda_model: modelo previamente creado
    #vect_class: nombre de la clase countvectorizer() usada antes
    
    df_topic_keywords = pd.DataFrame(lda_model.components_)

    # Assign Column and Index
    topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]
    df_topic_keywords.columns = vect_class.get_feature_names()
    df_topic_keywords.index = topicnames
    
    return df_topic_keywords

def show_topics(vect_class, lda_model, n_words):
    
    df_topic_keywords=topic_keyword(lda_model, vect_class)
    keywords = np.array(vect_class.get_feature_names())
    topic_keywords = []
    
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
     
    df_topic_keywords = pd.DataFrame(topic_keywords)
    df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
    df_topic_keywords=df_topic_keywords.T
    
    return df_topic_keywords

def final_result(obs_index,lda_model,info_vec):
    #obs_index: dataset.index
    #lda_model: bestldamodel
    #info_vec: matrix token-frequency for info
    
    #hacemos coincidir los índices
    final=obs_index
    final.index.name='Num trace'
    final=final.reset_index()
    
    #creamos el dataframe token-trace
    df_topic_dc=topic_document(lda_model,info_vec)
    #concat
    final_result=pd.concat([final, df_topic_dc], axis=1,)
    return final_result


# In[1]:


#LDA Model, inputs:dictionary,bow,num topics
def model_lda(dicc,bow,ntopic):
    lda =models.LdaModel(corpus=bow, id2word=dicc, 
               num_topics=ntopic, random_state=42, 
               chunksize=100, passes=50, alpha='auto')
    
    return lda  


# In[2]:


def traintestdata(obs_serie,num):
    X_train, X_test = train_test_split(obs_serie,  test_size=num, random_state=42)
    return X_train,X_test


# In[3]:


#pyLDAvis: Python library for interactive topic model visualization
def vis_lda(lda,corpus_lda,dictionary):
    pyLDAvis.enable_notebook()
    panel = pyLDAvis.gensim_models.prepare(lda, corpus_lda, dictionary, mds='tsne')
    return panel


# In[5]:


def compute_coherence_values(dictionary, corpu, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        
        model=models.LdaModel(corpus=corpu, id2word=dictionary, num_topics=num_topics, random_state=42,chunksize=100, passes=50, alpha='auto')
        
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[7]:


def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 10:31:51 2022

@author: arsen
"""


import time
import pickle 
from random import sample, seed, choices
from gensim.models import Word2Vec, phrases 
import cython
from gensim.test.utils import datapath
from gensim.models.phrases import Phraser
from nltk.tokenize import word_tokenize


###### Train Bigrammers on a Sample of the Data (do just once per time window) ####

curryear=1992

sampled_articles_for_bigrammer=[] #list of sentences, drawn from a sample of articles


for i in [curryear, curryear+1, curryear+2]:
    try:
        file = open('C:/Users/arsen/Dropbox/R01DiseaseStigma/LexisNexisAPI_DataCollection/RawData_NotSynced_To_Desktop/NData_' + str(i) + '/all' + str(i)+ 'bodytexts_regexeddisamb_listofarticles', 'rb') #do earliest of the three years to latest
    except:
        file = open('C:/Users/arsen/Dropbox/R01DiseaseStigma/LexisNexisAPI_DataCollection/RawData_NotSynced_To_Desktop/ContempData_' + str(i) + '/all' + str(i)+ 'bodytexts_regexeddisamb_listofarticles', 'rb') #do earliest of the three years to latest
    tfile_split= pickle.load(file)
    file.close()  
    samp_n= round(.75*len(tfile_split)) #sample size to sample the articles (next line) since there are so many articles
    tfile_split= sample(tfile_split, samp_n) 

    tfile_split= [i.split(' SENTENCEBOUNDARYHERE ') for i in tfile_split] #split the articles into sentences
    for article in tfile_split:
        for sentences_list in article:
            sentences= sentences_list.split()
            sampled_articles_for_bigrammer.append(sentences) #just adding the sentences themselves, using extend rather than append


bigram_transformer = phrases.Phrases(sampled_articles_for_bigrammer, min_count=50, threshold=12) #to TRAIN the bigrammer, input your "sentences" object with the cleaned text data. 
# min count: Ignore all words and bigrams with total collected count lower than this value.
bigram_transformer.save("C:/Users/arsen/Dropbox/R01DiseaseStigma/LexisNexisNews_Data_Modeling/bigrammer_" + str(curryear) + "_" + str(curryear+2))        


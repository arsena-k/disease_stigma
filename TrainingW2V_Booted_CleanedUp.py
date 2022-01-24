# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:34:02 2020

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

def  write_booted_txt(cyear, seed_no):
    #function to write a text file with bootstrapped data for a given time window for w2v model training
    all_articles=[]
    for i in [cyear, cyear+1, cyear+2]:
        try:
            file = open('C:/Users/arsen/Dropbox/R01DiseaseStigma/LexisNexisAPI_DataCollection/RawData_NotSynced_To_Desktop/NData_' + str(i) + '/all' + str(i)+ 'bodytexts_regexeddisamb_listofarticles', 'rb') #do earliest of the three years to latest
            tfile_split= pickle.load(file) #this is a list where each item in list is an article
            file.close()   
            all_articles.extend(tfile_split)
        except:
            file = open('C:/Users/arsen/Dropbox/R01DiseaseStigma/LexisNexisAPI_DataCollection/RawData_NotSynced_To_Desktop/ContempData_' + str(i) + '/all' + str(i)+ 'bodytexts_regexeddisamb_listofarticles', 'rb') #do earliest of the three years to latest
            tfile_split= pickle.load(file) #this is a list where each item in list is an article
            file.close()  
            all_articles.extend(tfile_split)
    
        
    with open('C:/Users/arsen/Dropbox/R01DiseaseStigma/LexisNexisNews_Data_Modeling/allarticles_tabsep_' + str(cyear) + "_" + str(cyear+2) + 'tempboot', 'w', encoding='utf-8') as f:
        seed(seed_no)
        
        all_articles = choices(all_articles,k= len(all_articles)) #resampled list for a bootstrapped version of the news in this year
        
        
        for article in all_articles:
            sentences_list= article.split(' SENTENCEBOUNDARYHERE ') #split the articles into sentences
    
            for sent in sentences_list:
                #sent= sent.split()
                f.write( sent)
                f.write("\n")



#from https://stackoverflow.com/questions/46421771/text-processing-word2vec-training-after-phrase-detection-bigram-model
#and from: #from: https://stackoverflow.com/questions/55086734/train-gensim-word2vec-using-large-txt-file
class SentenceIterator:   
    def __init__(self, filepath): 
        self.filepath = filepath

    def __iter__(self): 
        for line in open(self.filepath, "r", encoding="utf-8" ): 
            yield word_tokenize(line.rstrip('\n'))

           
class PhrasingIterable(object):
    def __init__(self, phrasifier, texts):
        self.phrasifier, self.texts = phrasifier, texts
    def __iter__(self):
        return iter(self.phrasifier[self.texts])





############# W2V Training (Train 25 models per time window)


curryear=1992

bigram_transformer= Phraser.load("C:/Users/arsen/Dropbox/R01DiseaseStigma/LexisNexisNews_Data_Modeling/bigrammer_" + str(curryear) + "_" + str(curryear+2))        

for boot in list(range(0,25)): 
    
    write_booted_txt(curryear,boot)
    
    sentences = SentenceIterator('C:/Users/arsen/Dropbox/R01DiseaseStigma/LexisNexisNews_Data_Modeling/allarticles_tabsep_' + str(curryear) + "_" + str(curryear+2) + 'tempboot') 
    corpus = PhrasingIterable(bigram_transformer, sentences)
    time.sleep(120)
    model1 = Word2Vec(corpus, workers=5, window=10, sg=0,size=300, min_count=50, iter=3) #added in max vocab size last, consider upping the min vocab word count. 
    model1.init_sims(replace=True) #Precompute L2-normalized vectors. If replace is set to TRUE, forget the original vectors and only keep the normalized ones. Saves lots of memory, but can't continue to train the model.
    model1.save("C:/Users/arsen/Dropbox/R01DiseaseStigma/LexisNexisNews_Data_Modeling/BootstrappedModels/CBOW_300d__win10_min50_iter3_" + str(curryear) + "_" + str(curryear+2) + '_boot' + str(boot)) #save model for later use! change the name to something to remember the hyperparameters you trained it with
    time.sleep(120)


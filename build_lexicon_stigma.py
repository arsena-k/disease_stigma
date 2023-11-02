# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 13:23:22 2019

@author: Alina Arseniev
"""


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from itertools import combinations
from random import seed, sample


#this function cleans up a word list (or single word), so that only words in the w2v model are included
def clean_words(word_list, w2vmodel, returnNA, min_count=1): #by default, doesn't return words not in the vocab
    assert type(word_list)== list, "Enter words as a list"
    cleaned_list= []
    for i in word_list:
        if returnNA==False:
            try:
                w2vmodel.wv[i]
                if w2vmodel.wv.vocab[i].count >= min_count: #skip this word if it is not in the model at least min count times
                    cleaned_list.append(i)
            except KeyError: #skip this word if it is not in the model
                continue
        elif returnNA==True:
            try:
                w2vmodel.wv[i]
                if w2vmodel.wv.vocab[i].count >= min_count:
                    cleaned_list.append(i)
                else:
                    cleaned_list.append(np.nan)
            except KeyError:
                cleaned_list.append(np.nan) #add nan if this word if it is not in the model 
                continue
    return cleaned_list
    
 
    
#this is a class that just keeps (cleaned) vocabulary to build a dimension (or cluster) 
#It cleans the vocab we want to use to build a dimension, only vocabulary that is actually in the data. 
#It then samples training and testing sets from the inputting lexicon

class dimension_lexicon:
    def __init__(self,  w2vmodel,init_pos_train_words, init_neg_train_words=None,  init_pos_test_words=None, init_neg_test_words=None, test_size=0, min_count=50): #test_size is of EACH side!, and only applicable for dimensions (i.e., when we are looking at pos and neg training words)
        self.w2vmodel= w2vmodel
        self.init_pos_train_words= init_pos_train_words
        self.init_neg_train_words= init_neg_train_words
        self.init_pos_test_words= init_pos_train_words
        self.init_neg_test_words= init_neg_train_words
        self.test_size= test_size  #must be either positive and smaller than the number of samples or a float in the (0, 1) range
        
        #cluster
        if self.init_neg_train_words== None: 
            self.pos_train= clean_words(self.init_pos_train_words, self.w2vmodel, returnNA=False,  min_count=min_count)
        
        #dimension, but don't want to look at testing words
        elif self.init_neg_test_words == None and self.init_pos_test_words == None: 
            self.pos_train= clean_words(self.init_pos_train_words, self.w2vmodel, returnNA=False,  min_count=min_count)
            self.neg_train= clean_words(self.init_neg_train_words, self.w2vmodel, returnNA=False,  min_count=min_count)
            
        #dimension, have pre-specficied testing sets of words 
        elif self.init_neg_test_words != None and self.init_pos_test_words != None:
            self.pos_train= clean_words(self.init_pos_train_words, self.w2vmodel, returnNA=False,  min_count=min_count)
            self.neg_train= clean_words(self.init_neg_train_words, self.w2vmodel, returnNA=False,  min_count=min_count)
            
            self.pos_test= clean_words(self.init_pos_test_words, self.w2vmodel, returnNA=False,  min_count=min_count)
            self.neg_test= clean_words(self.init_neg_test_words, self.w2vmodel, returnNA=False,  min_count=min_count)
        
        
        #dimension, have training words and want to separate out sample for testing using a random sample
        elif test_size > 0:
            cleaned_pos= clean_words(self.init_pos_train_words, self.w2vmodel, returnNA=False,  min_count=min_count)
            cleaned_neg= clean_words(self.init_neg_train_words, self.w2vmodel, returnNA=False,  min_count=min_count)
            
            if len(cleaned_pos) != len(cleaned_neg): 
                min_group_n= min(len(cleaned_pos), len(cleaned_neg))
                
                self.pos_train, self.pos_test = train_test_split(cleaned_pos , test_size=round((min_group_n * self.test_size)), random_state=42) #NOTE there may be different numbers of pos/neg  resulting sizes for test and train, if the training words in vocab are not the same sizes between pos and neg
                self.neg_train, self.neg_test = train_test_split(cleaned_neg , test_size=round((min_group_n * self.test_size)), random_state=42) #NOTE there may be different numbers of pos/neg  resulting sizes for test and train, if the training words in vocab are not the same sizes between pos and neg
    
            else:
                self.pos_train, self.pos_test = train_test_split(cleaned_pos , test_size=self.test_size, random_state=42)
                self.neg_train, self.neg_test = train_test_split(cleaned_neg , test_size=self.test_size, random_state=42)
        
        
            
 








        
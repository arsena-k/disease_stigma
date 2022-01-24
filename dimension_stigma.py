# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 09:41:20 2019

@author: Alina Arseniev
"""
import numpy as np
import copy
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec, KeyedVectors
from statistics import mode, mean, stdev, median
from sklearn.model_selection import KFold 
from sklearn.metrics.pairwise import cosine_similarity


def calc_wordlist_mean(wordlist, w2vmodel):
    wordlist= [w2vmodel.wv[i] for i in wordlist]
    meanvec = np.mean(wordlist,0) 
    meanvec= preprocessing.normalize(meanvec.reshape(1,-1), norm='l2') #ensure normalized to len 1
    meanvec= meanvec.reshape(w2vmodel.vector_size,)  #now will work with gensim similarity fcns
    return(meanvec)
   
    
class dimension: 
    def __init__(self, semantic_direction, method):
        self.semantic_direction= semantic_direction #this is another class, made with code in build_lexicon.py
        self.method= method #larsen or cluster
        self.w2vmodel= semantic_direction.w2vmodel
        assert self.method in ('larsen', 'cluster'), "Select one method: 'larsen' or 'cluster'"

        #get difference between two sets vectors
    def calc_dim_larsen(self): #VERIFY THIS WORKS
        diffvec= calc_wordlist_mean(self.semantic_direction.pos_train, self.w2vmodel) - calc_wordlist_mean(self.semantic_direction.neg_train, self.w2vmodel)
        diffvec= preprocessing.normalize(diffvec.reshape(1,-1), norm='l2')
        diffvec= diffvec.reshape(self.w2vmodel.vector_size,) #now will work with gensim similarity fcns
        return(diffvec)
        #return cossims between the found vector and some new word(s), and choose returnNAs if you still want to return words even if NAs 
    
    def dimensionvec(self):  
        if self.method =='larsen':
            return self.calc_dim_larsen() #this is the dimension according to the larsen method
        elif self.method =='cluster':
            return calc_wordlist_mean(self.semantic_direction.pos_train, self.w2vmodel) #this is just a cluster
    
    def cos_sim(self,inputwords, returnNAs): 
        assert type(inputwords)==list, "Enter word(s) as a list, e.g., ['word']"
        interesting_dim=self.dimensionvec().reshape(1,-1) 
        cossims= []
        for i in np.array(inputwords):
            if i=='nan' and returnNAs==True:
                cossims.append(np.nan)
            elif i!='nan':
                try:
                    cossims.append(cosine_similarity(self.w2vmodel.wv[i].reshape(1,-1),interesting_dim)[0][0])
                except KeyError:
                    if returnNAs==True:
                        cossims.append(np.nan)
                    continue
        return(cossims)

    def trainaccuracy(self): 
        assert self.method == "larsen", "Accuracy is only applicable to dimensions, not clusters"
        true_class=[]
        cossim_vec= []
        predicted_class=[]
        cossim_vec.extend(self.cos_sim(self.semantic_direction.pos_train, returnNAs= False))
        cossim_vec.extend(self.cos_sim(self.semantic_direction.neg_train, returnNAs=False))
        
        for i in self.semantic_direction.pos_train:
            true_class.append(1)
        for i in self.semantic_direction.neg_train:
            true_class.append(0)
        for i in cossim_vec:
            if float(i) > 0:
                predicted_class.append(1)
            if float(i) <0:
                predicted_class.append(0)
        accuracy= accuracy_score(true_class, predicted_class)  
        accuracy_n= accuracy_score(true_class, predicted_class, normalize=False)  
        return(accuracy, accuracy_n, true_class, predicted_class, cossim_vec) #note: this will not, by default, include any training words not in the vocabulary (i.e. don't use this across bootstrapped models, just useful for one model)
    
    def testaccuracy(self):
        assert self.method == "larsen", "Accuracy is only applicable to dimensions, not clusters"
        true_class=[]
        cossim_vec= []
        predicted_class=[]
        cossim_vec.extend(self.cos_sim(self.semantic_direction.pos_test, returnNAs=False))
        cossim_vec.extend(self.cos_sim(self.semantic_direction.neg_test, returnNAs=False))
        
        for i in self.semantic_direction.pos_test:
            true_class.append(1)
        for i in self.semantic_direction.neg_test:
            true_class.append(0)
        for i in cossim_vec:
            if float(i) > 0:
                predicted_class.append(1)
            if float(i) <0:
                predicted_class.append(0)
        
        accuracy= accuracy_score(true_class, predicted_class)  
        accuracy_n= accuracy_score(true_class, predicted_class, normalize=False)  
        return(accuracy, accuracy_n, true_class, predicted_class, cossim_vec) #note: this will not, by default, include any testing words not in the vocabulary (i.e. don't use this across bootstrapped models, just useful for one model)


def kfold_dim(semantic_direction,method='larsen', splits= 10): 
    min_group_n= len(min([semantic_direction.pos_train, semantic_direction.neg_train], key=len))
    min_group= semantic_direction.pos_train if len(semantic_direction.pos_train)== min_group_n else semantic_direction.neg_train #splits will computed based on the smaller N of pos and neg train
    kf= KFold(n_splits=splits, shuffle=True)  
    testaccy=[]
    trainaccy=[]
    trainaccy_n=[]
    testaccy_n=[]
    for train_index, test_index in kf.split(np.array(min_group)): #kf.split(np.array(semantic_direction.pos_train)):
        
        semantic_directionk = copy.deepcopy(semantic_direction)
        
        semantic_directionk.pos_test = list(np.array(semantic_directionk.pos_train)[test_index]) #must do test indices first since modifying training words next 
        semantic_directionk.neg_test = list(np.array(semantic_directionk.neg_train)[test_index])
   
        semantic_directionk.pos_train = list(np.array(semantic_directionk.pos_train)[train_index]) #kfold in sklearn only works for arrays
        semantic_directionk.neg_train = list(np.array(semantic_directionk.neg_train)[train_index])
   
        larsen_k= dimension(semantic_directionk, method='larsen') 
        #print("Train/Hold-out Subset Accuracies:")
        a,b,c,d,e= larsen_k.trainaccuracy()
        f,g,h,j,k = larsen_k.testaccuracy()
        trainaccy.append(a)
        trainaccy_n.append(b)
        testaccy.append(f)
        testaccy_n.append(g)
    return(mean(trainaccy_n), mean(trainaccy),  mean(testaccy_n), mean(testaccy))
    #print("Each training subset size (each pole): " + str(len(train_index)), "Each hold-out subset size (each pole): " + str(len(test_index)))
    #print('\033[1m' +'Mean (%,N), SD, Min Accuracy across Training Subsets: '  + '\033[0m'+ str(round(mean(trainaccy), 2)), str(mean(trainaccy_n)), str(round(stdev(trainaccy),2)),str(round(min(trainaccy),2)) )
    #print('\033[1m' +  'Mean (%,N), SD, Min Accuracy across Held-Out Subsets: ' + '\033[0m'+ str(round(mean(testaccy),2)), str(mean(testaccy_n)), str(round(stdev(testaccy),2)), str(round(min(testaccy),2)))
   


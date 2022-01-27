# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 08:37:28 2021

@author: arsen
"""




import os
import numpy as np
#import spacy #Alina run in in spacy21 conda env
import pandas as pd
from gensim.models import Word2Vec

os.chdir('C:/Users/arsen/Dropbox/R01DiseaseStigma/Analyses/') 

import build_lexicon_stigma 
import word_lists_stigma #imported mainly as a double check that this is in the working directory
import dimension_stigma
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr





#DIMENSIONS
lexicon= pd.read_csv('C:/Users/arsen/Dropbox/R01DiseaseStigma/Lexicon/Stigma_WordLists.csv')
lexicon= lexicon[lexicon['Removed']!='remove']

dangerouswords= lexicon.loc[(lexicon['WhichPole'] == 'dangerous')]['Term'].str.lower().tolist() 
safewords= lexicon.loc[(lexicon['WhichPole'] == 'safe')]['Term'].str.lower().tolist() 

disgustingwords= lexicon.loc[(lexicon['WhichPole'] == 'disgusting')]['Term'].str.lower().tolist() 
enticingwords= lexicon.loc[(lexicon['WhichPole'] == 'enticing')]['Term'].str.lower().tolist() 

purewords= lexicon.loc[(lexicon['WhichPole'] == 'pure')]['Term'].str.lower().tolist() 
impurewords= lexicon.loc[(lexicon['WhichPole'] == 'impure')]['Term'].str.lower().tolist() 
       
negtraits= pd.read_csv('C:/Users/arsen/Dropbox/R01DiseaseStigma/Lexicon/Personality traits/updated_personality_trait_list.csv')
negtraits['Adjective']= negtraits['Adjective'].str.lower()
negtraits['Adjective']= negtraits['Adjective'].str.strip()
negtraits= negtraits.drop_duplicates(subset=  'Adjective')
negwords= negtraits[negtraits['Sentiment']=='neg']['Adjective'].tolist()
poswords= negtraits[negtraits['Sentiment']=='pos']['Adjective'].tolist()
        
        
         
    
    
#PREP LISTS WITH RESULTS TO POPULATE

train_accuracy_N_danger =[]
train_accuracy_percent_danger=[]
holdout_accuracy_N_danger=[]
holdout_accuracy_percent_danger=[]

train_accuracy_N_disgust =[]
train_accuracy_percent_disgust=[]
holdout_accuracy_N_disgust=[]
holdout_accuracy_percent_disgust=[]

train_accuracy_N_purity =[]
train_accuracy_percent_purity=[]
holdout_accuracy_N_purity=[]
holdout_accuracy_percent_purity=[]

train_accuracy_N_negpos=[]
train_accuracy_percent_negpos=[]
holdout_accuracy_N_negpos=[]
holdout_accuracy_percent_negpos=[]

cossim_pure_danger=[]
cossim_pure_disgust=[]
cossim_pure_negpos=[]
cossim_danger_disgust=[]
cossim_negpos_disgust=[]
cossim_negpos_danger=[]

mostsim_danger=[]
leastsim_danger=[]
mostsim_purity=[]
leastsim_purity=[]
mostsim_negpos=[]
leastsim_negpos=[]
mostsim_disgust=[]
leastsim_disgust=[]




    
for yr1 in   [1980, 1983, 1986, 1989, 1992, 1995, 1998, 2001, 2004, 2007, 2010, 2013, 2016]:
    print('PROCESSING MODEL FOR YEAR:', yr1)
    yr3=yr1 + 2
    
    #load in boot0 model for this year
    currentmodel= Word2Vec.load('C:/Users/arsen/Dropbox/R01DiseaseStigma/LexisNexisNews_Data_Modeling/BootstrappedModels/' + str(yr1) + '_' + str(yr3) + '/CBOW_300d__win10_min50_iter3_'+ str(yr1)+ '_' + str(yr3) + "_boot0") #load in desired model        

    #first combine the two poles for each dimension
    dangerwords=  build_lexicon_stigma.dimension_lexicon( currentmodel,  dangerouswords,  safewords)
    disgustwords=  build_lexicon_stigma.dimension_lexicon( currentmodel,  disgustingwords,  enticingwords)    
    puritywords=  build_lexicon_stigma.dimension_lexicon( currentmodel,  impurewords, purewords)
    negposwords=  build_lexicon_stigma.dimension_lexicon( currentmodel,  negwords,  poswords)
    
    
    #second, create the dimension
    danger= dimension_stigma.dimension(dangerwords,'larsen') 
    disgust= dimension_stigma.dimension(disgustwords,'larsen') 
    purity= dimension_stigma.dimension(puritywords,'larsen') 
    negpos= dimension_stigma.dimension(negposwords,'larsen')    

    #third, get the cross validation accuracy for each of the four dimensions
    
    dimtemp= dimension_stigma.kfold_dim(disgustwords) 
    train_accuracy_N_disgust.append(dimtemp[0])
    train_accuracy_percent_disgust.append(dimtemp[1])
    holdout_accuracy_N_disgust.append(dimtemp[2])
    holdout_accuracy_percent_disgust.append(dimtemp[3])
    
    dimtemp=dimension_stigma.kfold_dim(puritywords)  
    train_accuracy_N_purity.append(dimtemp[0])
    train_accuracy_percent_purity.append(dimtemp[1])
    holdout_accuracy_N_purity.append(dimtemp[2])
    holdout_accuracy_percent_purity.append(dimtemp[3])

    dimtemp=dimension_stigma.kfold_dim(dangerwords) 
    train_accuracy_N_danger.append(dimtemp[0])
    train_accuracy_percent_danger.append(dimtemp[1])
    holdout_accuracy_N_danger.append(dimtemp[2])
    holdout_accuracy_percent_danger.append(dimtemp[3])

    dimtemp=dimension_stigma.kfold_dim(negposwords)
    train_accuracy_N_negpos.append(dimtemp[0])
    train_accuracy_percent_negpos.append(dimtemp[1])
    holdout_accuracy_N_negpos.append(dimtemp[2])
    holdout_accuracy_percent_negpos.append(dimtemp[3])



    #fourth, compute cosine similarities between vectors corresponding to the four dimensions

    cossim_pure_danger.append(cosine_similarity(purity.dimensionvec().reshape(1,-1), danger.dimensionvec().reshape(1,-1)))
    cossim_pure_disgust.append(cosine_similarity(purity.dimensionvec().reshape(1,-1), disgust.dimensionvec().reshape(1,-1)))
    cossim_pure_negpos.append(cosine_similarity(purity.dimensionvec().reshape(1,-1), negpos.dimensionvec().reshape(1,-1)))
    cossim_danger_disgust.append(cosine_similarity(danger.dimensionvec().reshape(1,-1), disgust.dimensionvec().reshape(1,-1)))
    cossim_negpos_disgust.append(cosine_similarity(negpos.dimensionvec().reshape(1,-1), disgust.dimensionvec().reshape(1,-1)))
    cossim_negpos_danger.append(cosine_similarity(negpos.dimensionvec().reshape(1,-1), danger.dimensionvec().reshape(1,-1)))
    
    '''
    #fifth, compute correlation between dimensions (double checking that this is same as correlation)
    
    corr_pure_danger.append(pearsonr(purity.dimensionvec(), danger.dimensionvec())[0])
    corr_pure_disgust.append(pearsonr(purity.dimensionvec(), disgust.dimensionvec())[0])
    corr_pure_negpos.append(pearsonr(purity.dimensionvec(), negpos.dimensionvec())[0])
    corr_danger_disgust.append(pearsonr(danger.dimensionvec(), disgust.dimensionvec())[0])
    corr_negpos_disgust.append(pearsonr(negpos.dimensionvec(), disgust.dimensionvec())[0])
    corr_negpos_danger.append(pearsonr(negpos.dimensionvec(), danger.dimensionvec())[0])
    '''
    #sixth, get least/most similar words
    
    mostsim_danger.append( currentmodel.wv.similar_by_vector(danger.dimensionvec(), topn=10))
    leastsim_danger.append( currentmodel.wv.similar_by_vector(-danger.dimensionvec(), topn=10))
    mostsim_purity.append( currentmodel.wv.similar_by_vector(purity.dimensionvec(), topn=10))
    leastsim_purity.append( currentmodel.wv.similar_by_vector(-purity.dimensionvec(), topn=10))
    mostsim_disgust.append( currentmodel.wv.similar_by_vector(disgust.dimensionvec(), topn=10))
    leastsim_disgust.append( currentmodel.wv.similar_by_vector(-disgust.dimensionvec(), topn=10))
    mostsim_negpos.append( currentmodel.wv.similar_by_vector(negpos.dimensionvec(), topn=10))
    leastsim_negpos.append( currentmodel.wv.similar_by_vector(-negpos.dimensionvec(), topn=10))



# Now, print out desired results

for i in [cossim_pure_danger, cossim_pure_disgust, cossim_pure_negpos,cossim_danger_disgust,
          cossim_danger_disgust, cossim_negpos_disgust, cossim_negpos_danger]:
    print(round(np.mean(i), 2))
    print(round(np.std(i), 2))
    print("next")
    











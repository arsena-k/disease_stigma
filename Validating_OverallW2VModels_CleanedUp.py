# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 11:08:03 2022

@author: arsen
"""


import os
from gensim.models import Word2Vec
from gensim.test.utils import datapath

os.chdir('C:/Users/arsen/Dropbox/R01DiseaseStigma/Analyses/') 

#Note, the Google Analogy Test requires "questions_words_pasted.txt" developed by Mikolov et al: https://aclweb.org/aclwiki/Google_analogy_test_set_(State_of_the_art)





# Load in desired word2vec model

model1= Word2Vec.load('C:/Users/arsen/Dropbox/R01DiseaseStigma/LexisNexisNews_Data_Modeling/Old_EmbeddingsTrainedonAllData_ForHyperparameterSelection/Cleaned3yrDataModels_OLD/FinalModels_AlldataOnlyForHyperparamters/CBOW_300d__win10_min50_iter5_1989_1991') 




# TEST WORD2VEC MODEL PERFORMANCE ON WORDSIM-353 TEST

model1.wv.evaluate_word_pairs(datapath("wordsim353.tsv"))




# TEST WORD2VEC MODEL PERFORMANCE ON GOOGLE ANALOGY TEST

#Requires questions_words_pasted.txt.
acc1, acc2 = model1.wv.evaluate_word_analogies('questions_words_pasted.txt')
acc2_labels= ['capital-common-countries', 'capital-world', 'money', 'US_capitals', 'family', 'adj_to_adverbs', 'opposites', 'comparative', 'superlative','present_participle', 'nationality', 'past_tense', 'plural', 'plural_verbs', 'total accuracy']

accuracy_tracker=[]
for i in range(0, len(acc2)):
    sum_corr = len(acc2[i]['correct'])
    sum_incorr = len(acc2[i]['incorrect'])
    total = sum_corr + sum_incorr
    print("Accuracy on " + str(acc2_labels[i]) + ": "  + str(float(sum_corr)/(total)))
    accuracy_tracker.append(float(sum_corr)/(total))


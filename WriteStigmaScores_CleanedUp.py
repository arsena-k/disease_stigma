# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 10:47:20 2022

@author: arsen
"""


import pandas as pd
import os
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
from sklearn.preprocessing import normalize

os.chdir('C:/Users/arsen/Dropbox/R01DiseaseStigma/Analyses/') 
import build_lexicon_stigma 
#import word_lists_stigma #imported as a double check that this is in the working directory
import dimension_stigma


def fold_word(target, second, wvmodel): #following the convex combination example on wiki
    weight_target = wvmodel.wv.vocab[target].count / (wvmodel.wv.vocab[target].count  + wvmodel.wv.vocab[second].count) 
    weight_second = wvmodel.wv.vocab[second].count / (wvmodel.wv.vocab[target].count  + wvmodel.wv.vocab[second].count) 
    weighted_wv= (weight_target* normalize(wvmodel.wv[target].reshape(1,-1))  ) + (weight_second* normalize(wvmodel.wv[second].reshape(1,-1)) ) 
    return ( normalize(weighted_wv) )


################ Load in keyword lists from the file "Stigma_WordLists.csv")

lexicon= pd.read_csv('C:/Users/arsen/Dropbox/R01DiseaseStigma/Lexicon/Stigma_WordLists.csv')
lexicon= lexicon[lexicon['Removed']!='remove']

dangerouswords= lexicon.loc[(lexicon['WhichPole'] == 'dangerous')]['Term'].str.lower().tolist() 
safewords= lexicon.loc[(lexicon['WhichPole'] == 'safe')]['Term'].str.lower().tolist() 

disgustingwords= lexicon.loc[(lexicon['WhichPole'] == 'disgusting')]['Term'].str.lower().tolist() 
enticingwords= lexicon.loc[(lexicon['WhichPole'] == 'enticing')]['Term'].str.lower().tolist() 

moralwords= lexicon.loc[(lexicon['WhichPole'] == 'moral')]['Term'].str.lower().tolist() 
immoralwords= lexicon.loc[(lexicon['WhichPole'] == 'immoral')]['Term'].str.lower().tolist() 

purewords= lexicon.loc[(lexicon['WhichPole'] == 'pure')]['Term'].str.lower().tolist() 
impurewords= lexicon.loc[(lexicon['WhichPole'] == 'impure')]['Term'].str.lower().tolist() 
    
medwords= lexicon.loc[(lexicon['Concept_Term_Represents'] == 'medicalization')]['Term'].str.lower().tolist() 


################ CSV with Danger Scores for Each Diseases from Each Model in Each Time Period

for yr1 in [1980, 1983, 1986, 1989, 1992, 1995, 1998, 2001, 2004, 2007, 2010, 2013, 2016]:
    diseases= pd.read_csv('C:/Users/arsen/Dropbox/R01DiseaseStigma/Disease_list_5.12.20_uncorrupted.csv')
    diseases = diseases[ diseases['Plot'] == 'Yes' ]
    diseases = diseases.drop_duplicates(subset=['Reconciled_Name'])
    diseases = diseases[['PlottingGroup', 'Reconciled_Name']]
    yr3=yr1 + 2
    for bootnum in list(range(0,25)):
        
        currentmodel1= KeyedVectors.load('C:/Users/arsen/Dropbox/R01DiseaseStigma/LexisNexisNews_Data_Modeling/BootstrappedModels/' + str(yr1) + '_' + str(yr3) + '/CBOW_300d__win10_min50_iter3_'+ str(yr1)+ '_' + str(yr3) + "_boot" + str(bootnum)) #load in desired model

        #adding in three post-hoc word-vectors. This is TEMPORARY and not changing the underlying model. 
        epilepsy_folded = fold_word('epilepsy', 'epileptic', currentmodel1)
        drug_addiction_folded = fold_word('drug_addiction', 'drug_addict', currentmodel1)
        obesity_folded = fold_word('obesity', 'obese', currentmodel1)
        
        currentmodel1.wv.add('epilepsy_folded', epilepsy_folded)
        currentmodel1.wv['drug_addiction_folded'] = drug_addiction_folded
        currentmodel1.wv['obesity_folded'] = obesity_folded
        
        #need to also update the count since this is used to exclude certain diseases
        currentmodel1.wv.vocab['epilepsy_folded'].count = currentmodel1.wv.vocab['epileptic'].count + currentmodel1.wv.vocab['epilepsy'].count
        currentmodel1.wv.vocab['drug_addiction_folded'].count = currentmodel1.wv.vocab['drug_addict'].count + currentmodel1.wv.vocab['drug_addiction'].count
        currentmodel1.wv.vocab['obesity_folded'].count = currentmodel1.wv.vocab['obese'].count + currentmodel1.wv.vocab['obesity'].count
        
        
        #Danger (Dimension)
        dangerwords_curr=  build_lexicon_stigma.dimension_lexicon( currentmodel1,  dangerouswords,  safewords)
        danger= dimension_stigma.dimension(dangerwords_curr,'larsen') 
        allwordssims_danger= danger.cos_sim(list(currentmodel1.wv.vocab),returnNAs=False)
        diseases['danger_score_stdized'+  '_' +str(bootnum)]= diseases['Reconciled_Name'].apply(lambda x: ((danger.cos_sim([str(x).lower()], returnNAs=True)[0])- np.mean(allwordssims_danger))/np.std(allwordssims_danger))
    diseases['Year']= [str(yr1)] * len(diseases)
    dimension_to_plot = 'danger'
    diseases = pd.melt(diseases[['Reconciled_Name', 'PlottingGroup', 'Year',  str(dimension_to_plot + '_score_stdized_0'), str(dimension_to_plot + '_score_stdized_1'), str(dimension_to_plot + '_score_stdized_2'), str(dimension_to_plot + '_score_stdized_3'),
                                 str(dimension_to_plot + '_score_stdized_4'), str(dimension_to_plot + '_score_stdized_5'), str(dimension_to_plot + '_score_stdized_6'), str(dimension_to_plot + '_score_stdized_7'), str(dimension_to_plot + '_score_stdized_8'), str(dimension_to_plot + '_score_stdized_9'), 
                                 str(dimension_to_plot + '_score_stdized_10'), str(dimension_to_plot + '_score_stdized_11'), str(dimension_to_plot + '_score_stdized_12'), str(dimension_to_plot + '_score_stdized_13'), str(dimension_to_plot + '_score_stdized_14'), 
                                  str(dimension_to_plot + '_score_stdized_15'), str(dimension_to_plot + '_score_stdized_16'), str(dimension_to_plot + '_score_stdized_17'), str(dimension_to_plot + '_score_stdized_18'), 
                                 str(dimension_to_plot + '_score_stdized_19'), str(dimension_to_plot + '_score_stdized_20'), str(dimension_to_plot + '_score_stdized_21'), str(dimension_to_plot + '_score_stdized_22'), str(dimension_to_plot + '_score_stdized_23'), 
                                 str(dimension_to_plot + '_score_stdized_24') ]],
                    id_vars=['Reconciled_Name', 'PlottingGroup', 'Year',],var_name='BootNumber', value_name= dimension_to_plot)
    diseases.to_csv('temp' + str(dimension_to_plot) + str(yr1) + '.csv')


################ CSV with Disgust Scores for Each Diseases from Each Model in Each Time Period

for yr1 in [1980, 1983, 1986, 1989, 1992, 1995, 1998, 2001, 2004, 2007, 2010, 2013, 2016]:
    diseases= pd.read_csv('C:/Users/arsen/Dropbox/R01DiseaseStigma/Disease_list_5.12.20_uncorrupted.csv')
    diseases = diseases[ diseases['Plot'] == 'Yes' ]
    diseases = diseases.drop_duplicates(subset=['Reconciled_Name'])
    diseases = diseases[['PlottingGroup', 'Reconciled_Name']]
    yr3=yr1 + 2
    for bootnum in list(range(0,25)):
        
        currentmodel1= KeyedVectors.load('C:/Users/arsen/Dropbox/R01DiseaseStigma/LexisNexisNews_Data_Modeling/BootstrappedModels/' + str(yr1) + '_' + str(yr3) + '/CBOW_300d__win10_min50_iter3_'+ str(yr1)+ '_' + str(yr3) + "_boot" + str(bootnum)) #load in desired model

        #adding in three post-hoc word-vectors. This is TEMPORARY and not changing the underlying model. 
        epilepsy_folded = fold_word('epilepsy', 'epileptic', currentmodel1)
        drug_addiction_folded = fold_word('drug_addiction', 'drug_addict', currentmodel1)
        obesity_folded = fold_word('obesity', 'obese', currentmodel1)
        
        currentmodel1.wv.add('epilepsy_folded', epilepsy_folded)
        currentmodel1.wv['drug_addiction_folded'] = drug_addiction_folded
        currentmodel1.wv['obesity_folded'] = obesity_folded
        
        #need to also update the count since this is used to exclude certain diseases
        currentmodel1.wv.vocab['epilepsy_folded'].count = currentmodel1.wv.vocab['epileptic'].count + currentmodel1.wv.vocab['epilepsy'].count
        currentmodel1.wv.vocab['drug_addiction_folded'].count = currentmodel1.wv.vocab['drug_addict'].count + currentmodel1.wv.vocab['drug_addiction'].count
        currentmodel1.wv.vocab['obesity_folded'].count = currentmodel1.wv.vocab['obese'].count + currentmodel1.wv.vocab['obesity'].count
        
        #Disgust (Dimension)
        disgustwords_curr=  build_lexicon_stigma.dimension_lexicon( currentmodel1,  disgustingwords,  enticingwords)
        disgust= dimension_stigma.dimension(disgustwords_curr,'larsen') 
        allwordssims_disgust= disgust.cos_sim(list(currentmodel1.wv.vocab),returnNAs=False)
        diseases['disgust_score_stdized'+  '_' +str(bootnum)]= diseases['Reconciled_Name'].apply(lambda x: ((disgust.cos_sim([str(x).lower()], returnNAs=True)[0])- np.mean(allwordssims_disgust))/np.std(allwordssims_disgust))      
    diseases['Year']= [str(yr1)] * len(diseases)
    dimension_to_plot = 'disgust'   
    diseases = pd.melt(diseases[['Reconciled_Name', 'PlottingGroup', 'Year',  str(dimension_to_plot + '_score_stdized_0'), str(dimension_to_plot + '_score_stdized_1'), str(dimension_to_plot + '_score_stdized_2'), str(dimension_to_plot + '_score_stdized_3'),
                                 str(dimension_to_plot + '_score_stdized_4'), str(dimension_to_plot + '_score_stdized_5'), str(dimension_to_plot + '_score_stdized_6'), str(dimension_to_plot + '_score_stdized_7'), str(dimension_to_plot + '_score_stdized_8'), str(dimension_to_plot + '_score_stdized_9'), 
                                 str(dimension_to_plot + '_score_stdized_10'), str(dimension_to_plot + '_score_stdized_11'), str(dimension_to_plot + '_score_stdized_12'), str(dimension_to_plot + '_score_stdized_13'), str(dimension_to_plot + '_score_stdized_14'), 
                                  str(dimension_to_plot + '_score_stdized_15'), str(dimension_to_plot + '_score_stdized_16'), str(dimension_to_plot + '_score_stdized_17'), str(dimension_to_plot + '_score_stdized_18'), 
                                 str(dimension_to_plot + '_score_stdized_19'), str(dimension_to_plot + '_score_stdized_20'), str(dimension_to_plot + '_score_stdized_21'), str(dimension_to_plot + '_score_stdized_22'), str(dimension_to_plot + '_score_stdized_23'), 
                                 str(dimension_to_plot + '_score_stdized_24') ]],
                    id_vars=['Reconciled_Name', 'PlottingGroup', 'Year',],var_name='BootNumber', value_name= dimension_to_plot)
    diseases.to_csv('temp' + str(dimension_to_plot) + str(yr1) + '.csv')

################ CSV with Immoral Scores for Each Diseases from Each Model in Each Time Period

for yr1 in [1980, 1983, 1986, 1989, 1992, 1995, 1998, 2001, 2004, 2007, 2010, 2013, 2016]:
    diseases= pd.read_csv('C:/Users/arsen/Dropbox/R01DiseaseStigma/Disease_list_5.12.20_uncorrupted.csv')
    diseases = diseases[ diseases['Plot'] == 'Yes' ]
    diseases = diseases.drop_duplicates(subset=['Reconciled_Name'])
    diseases = diseases[['PlottingGroup', 'Reconciled_Name']]
    yr3=yr1 + 2
    for bootnum in list(range(0,25)):
        currentmodel1= KeyedVectors.load('C:/Users/arsen/Dropbox/R01DiseaseStigma/LexisNexisNews_Data_Modeling/BootstrappedModels/' + str(yr1) + '_' + str(yr3) + '/CBOW_300d__win10_min50_iter3_'+ str(yr1)+ '_' + str(yr3) + "_boot" + str(bootnum)) #load in desired model

        #adding in three post-hoc word-vectors. This is TEMPORARY and not changing the underlying model. 
        epilepsy_folded = fold_word('epilepsy', 'epileptic', currentmodel1)
        drug_addiction_folded = fold_word('drug_addiction', 'drug_addict', currentmodel1)
        obesity_folded = fold_word('obesity', 'obese', currentmodel1)
        
        currentmodel1.wv.add('epilepsy_folded', epilepsy_folded)
        currentmodel1.wv['drug_addiction_folded'] = drug_addiction_folded
        currentmodel1.wv['obesity_folded'] = obesity_folded
        
        #need to also update the count since this is used to exclude certain diseases
        currentmodel1.wv.vocab['epilepsy_folded'].count = currentmodel1.wv.vocab['epileptic'].count + currentmodel1.wv.vocab['epilepsy'].count
        currentmodel1.wv.vocab['drug_addiction_folded'].count = currentmodel1.wv.vocab['drug_addict'].count + currentmodel1.wv.vocab['drug_addiction'].count
        currentmodel1.wv.vocab['obesity_folded'].count = currentmodel1.wv.vocab['obese'].count + currentmodel1.wv.vocab['obesity'].count
        
        #Immorality (Dimension)
        moralitywords_curr=  build_lexicon_stigma.dimension_lexicon( currentmodel1,  immoralwords,  moralwords)
        morality= dimension_stigma.dimension(moralitywords_curr,'larsen') 
        allwordssims_morality= morality.cos_sim(list(currentmodel1.wv.vocab),returnNAs=False)
        diseases['immorality_score_stdized'+  '_' +str(bootnum)]= diseases['Reconciled_Name'].apply(lambda x: ((morality.cos_sim([str(x).lower()], returnNAs=True)[0])- np.mean(allwordssims_morality))/np.std(allwordssims_morality))
    diseases['Year']= [str(yr1)] * len(diseases)
    dimension_to_plot = 'immorality' 
    diseases = pd.melt(diseases[['Reconciled_Name', 'PlottingGroup', 'Year',  str(dimension_to_plot + '_score_stdized_0'), str(dimension_to_plot + '_score_stdized_1'), str(dimension_to_plot + '_score_stdized_2'), str(dimension_to_plot + '_score_stdized_3'),
                                 str(dimension_to_plot + '_score_stdized_4'), str(dimension_to_plot + '_score_stdized_5'), str(dimension_to_plot + '_score_stdized_6'), str(dimension_to_plot + '_score_stdized_7'), str(dimension_to_plot + '_score_stdized_8'), str(dimension_to_plot + '_score_stdized_9'), 
                                 str(dimension_to_plot + '_score_stdized_10'), str(dimension_to_plot + '_score_stdized_11'), str(dimension_to_plot + '_score_stdized_12'), str(dimension_to_plot + '_score_stdized_13'), str(dimension_to_plot + '_score_stdized_14'), 
                                  str(dimension_to_plot + '_score_stdized_15'), str(dimension_to_plot + '_score_stdized_16'), str(dimension_to_plot + '_score_stdized_17'), str(dimension_to_plot + '_score_stdized_18'), 
                                 str(dimension_to_plot + '_score_stdized_19'), str(dimension_to_plot + '_score_stdized_20'), str(dimension_to_plot + '_score_stdized_21'), str(dimension_to_plot + '_score_stdized_22'), str(dimension_to_plot + '_score_stdized_23'), 
                                 str(dimension_to_plot + '_score_stdized_24') ]],
                    id_vars=['Reconciled_Name', 'PlottingGroup', 'Year',],var_name='BootNumber', value_name= dimension_to_plot)
    diseases.to_csv('temp' + str(dimension_to_plot) + str(yr1) + '.csv')

################ CSV with Impure Scores for Each Diseases from Each Model in Each Time Period

for yr1 in [1980, 1983, 1986, 1989, 1992, 1995, 1998, 2001, 2004, 2007, 2010, 2013, 2016]:
    diseases= pd.read_csv('C:/Users/arsen/Dropbox/R01DiseaseStigma/Disease_list_5.12.20_uncorrupted.csv')
    diseases = diseases[ diseases['Plot'] == 'Yes' ]
    diseases = diseases.drop_duplicates(subset=['Reconciled_Name'])
    diseases = diseases[['PlottingGroup', 'Reconciled_Name']]
    yr3=yr1 + 2
    for bootnum in list(range(0,25)):       
        currentmodel1= KeyedVectors.load('C:/Users/arsen/Dropbox/R01DiseaseStigma/LexisNexisNews_Data_Modeling/BootstrappedModels/' + str(yr1) + '_' + str(yr3) + '/CBOW_300d__win10_min50_iter3_'+ str(yr1)+ '_' + str(yr3) + "_boot" + str(bootnum)) #load in desired model

        #adding in three post-hoc word-vectors. This is TEMPORARY and not changing the underlying model. 
        epilepsy_folded = fold_word('epilepsy', 'epileptic', currentmodel1)
        drug_addiction_folded = fold_word('drug_addiction', 'drug_addict', currentmodel1)
        obesity_folded = fold_word('obesity', 'obese', currentmodel1)
        
        currentmodel1.wv.add('epilepsy_folded', epilepsy_folded)
        currentmodel1.wv['drug_addiction_folded'] = drug_addiction_folded
        currentmodel1.wv['obesity_folded'] = obesity_folded
        
        #need to also update the count since this is used to exclude certain diseases
        currentmodel1.wv.vocab['epilepsy_folded'].count = currentmodel1.wv.vocab['epileptic'].count + currentmodel1.wv.vocab['epilepsy'].count
        currentmodel1.wv.vocab['drug_addiction_folded'].count = currentmodel1.wv.vocab['drug_addict'].count + currentmodel1.wv.vocab['drug_addiction'].count
        currentmodel1.wv.vocab['obesity_folded'].count = currentmodel1.wv.vocab['obese'].count + currentmodel1.wv.vocab['obesity'].count
        
        #Impurity (Dimension)
        puritywords_curr=  build_lexicon_stigma.dimension_lexicon( currentmodel1,  impurewords, purewords)
        purity= dimension_stigma.dimension(puritywords_curr,'larsen') 
        allwordssims_purity= purity.cos_sim(list(currentmodel1.wv.vocab),returnNAs=False)
        diseases['impurity_score_stdized'+  '_' +str(bootnum)]= diseases['Reconciled_Name'].apply(lambda x: ((purity.cos_sim([str(x).lower()], returnNAs=True)[0])- np.mean(allwordssims_purity))/np.std(allwordssims_purity))
    diseases['Year']= [str(yr1)] * len(diseases)
    dimension_to_plot = 'impurity'
    diseases = pd.melt(diseases[['Reconciled_Name', 'PlottingGroup', 'Year',  str(dimension_to_plot + '_score_stdized_0'), str(dimension_to_plot + '_score_stdized_1'), str(dimension_to_plot + '_score_stdized_2'), str(dimension_to_plot + '_score_stdized_3'),
                                 str(dimension_to_plot + '_score_stdized_4'), str(dimension_to_plot + '_score_stdized_5'), str(dimension_to_plot + '_score_stdized_6'), str(dimension_to_plot + '_score_stdized_7'), str(dimension_to_plot + '_score_stdized_8'), str(dimension_to_plot + '_score_stdized_9'), 
                                 str(dimension_to_plot + '_score_stdized_10'), str(dimension_to_plot + '_score_stdized_11'), str(dimension_to_plot + '_score_stdized_12'), str(dimension_to_plot + '_score_stdized_13'), str(dimension_to_plot + '_score_stdized_14'), 
                                  str(dimension_to_plot + '_score_stdized_15'), str(dimension_to_plot + '_score_stdized_16'), str(dimension_to_plot + '_score_stdized_17'), str(dimension_to_plot + '_score_stdized_18'), 
                                 str(dimension_to_plot + '_score_stdized_19'), str(dimension_to_plot + '_score_stdized_20'), str(dimension_to_plot + '_score_stdized_21'), str(dimension_to_plot + '_score_stdized_22'), str(dimension_to_plot + '_score_stdized_23'), 
                                 str(dimension_to_plot + '_score_stdized_24') ]],
                    id_vars=['Reconciled_Name', 'PlottingGroup', 'Year',],var_name='BootNumber', value_name= dimension_to_plot)
    
    diseases.to_csv('temp' + str(dimension_to_plot) + str(yr1) + '.csv')
    
################ CSV with NegPosTraits Scores for Each Diseases from Each Model in Each Time Period

for yr1 in [1980, 1983, 1986, 1989, 1992, 1995, 1998, 2001, 2004, 2007, 2010, 2013, 2016]:
    diseases= pd.read_csv('C:/Users/arsen/Dropbox/R01DiseaseStigma/Disease_list_5.12.20_uncorrupted.csv')
    diseases = diseases[ diseases['Plot'] == 'Yes' ]
    diseases = diseases.drop_duplicates(subset=['Reconciled_Name'])
    diseases = diseases[['PlottingGroup', 'Reconciled_Name']]
    yr3=yr1 + 2
    for bootnum in list(range(0,25)):
        
        currentmodel1= KeyedVectors.load('C:/Users/arsen/Dropbox/R01DiseaseStigma/LexisNexisNews_Data_Modeling/BootstrappedModels/' + str(yr1) + '_' + str(yr3) + '/CBOW_300d__win10_min50_iter3_'+ str(yr1)+ '_' + str(yr3) + "_boot" + str(bootnum)) #load in desired model

        #adding in three post-hoc word-vectors. This is TEMPORARY and not changing the underlying model. 
        epilepsy_folded = fold_word('epilepsy', 'epileptic', currentmodel1)
        drug_addiction_folded = fold_word('drug_addiction', 'drug_addict', currentmodel1)
        obesity_folded = fold_word('obesity', 'obese', currentmodel1)
        
        currentmodel1.wv.add('epilepsy_folded', epilepsy_folded)
        currentmodel1.wv['drug_addiction_folded'] = drug_addiction_folded
        currentmodel1.wv['obesity_folded'] = obesity_folded
        
        #need to also update the count since this is used to exclude certain diseases
        currentmodel1.wv.vocab['epilepsy_folded'].count = currentmodel1.wv.vocab['epileptic'].count + currentmodel1.wv.vocab['epilepsy'].count
        currentmodel1.wv.vocab['drug_addiction_folded'].count = currentmodel1.wv.vocab['drug_addict'].count + currentmodel1.wv.vocab['drug_addiction'].count
        currentmodel1.wv.vocab['obesity_folded'].count = currentmodel1.wv.vocab['obese'].count + currentmodel1.wv.vocab['obesity'].count
        
        #NegTraits - PosTraits (Dimension)
        negtraits= pd.read_csv('C:/Users/arsen/Dropbox/R01DiseaseStigma/Lexicon/Personality traits/updated_personality_trait_list.csv')
        negtraits['Adjective']= negtraits['Adjective'].str.lower()
        negtraits['Adjective']= negtraits['Adjective'].str.strip()
        negtraits= negtraits.drop_duplicates(subset=  'Adjective')
        
        negwords= negtraits[negtraits['Sentiment']=='neg']['Adjective'].tolist()
        poswords= negtraits[negtraits['Sentiment']=='pos']['Adjective'].tolist()
        
        negposwords_curr=  build_lexicon_stigma.dimension_lexicon( currentmodel1,  negwords,  poswords)
        negpos= dimension_stigma.dimension(negposwords_curr,'larsen') 
        allwordssims_negpos= negpos.cos_sim(list(currentmodel1.wv.vocab),returnNAs=False)
        diseases['negpostraits_score_stdized'+  '_' +str(bootnum)]= diseases['Reconciled_Name'].apply(lambda x: ((negpos.cos_sim([str(x).lower()], returnNAs=True)[0])- np.mean(allwordssims_negpos))/np.std(allwordssims_negpos))

    diseases['Year']= [str(yr1)] * len(diseases)
    dimension_to_plot = 'negpostraits'    
    diseases = pd.melt(diseases[['Reconciled_Name', 'PlottingGroup', 'Year',  str(dimension_to_plot + '_score_stdized_0'), str(dimension_to_plot + '_score_stdized_1'), str(dimension_to_plot + '_score_stdized_2'), str(dimension_to_plot + '_score_stdized_3'),
                                 str(dimension_to_plot + '_score_stdized_4'), str(dimension_to_plot + '_score_stdized_5'), str(dimension_to_plot + '_score_stdized_6'), str(dimension_to_plot + '_score_stdized_7'), str(dimension_to_plot + '_score_stdized_8'), str(dimension_to_plot + '_score_stdized_9'), 
                                 str(dimension_to_plot + '_score_stdized_10'), str(dimension_to_plot + '_score_stdized_11'), str(dimension_to_plot + '_score_stdized_12'), str(dimension_to_plot + '_score_stdized_13'), str(dimension_to_plot + '_score_stdized_14'), 
                                  str(dimension_to_plot + '_score_stdized_15'), str(dimension_to_plot + '_score_stdized_16'), str(dimension_to_plot + '_score_stdized_17'), str(dimension_to_plot + '_score_stdized_18'), 
                                 str(dimension_to_plot + '_score_stdized_19'), str(dimension_to_plot + '_score_stdized_20'), str(dimension_to_plot + '_score_stdized_21'), str(dimension_to_plot + '_score_stdized_22'), str(dimension_to_plot + '_score_stdized_23'), 
                                 str(dimension_to_plot + '_score_stdized_24') ]],
                    id_vars=['Reconciled_Name', 'PlottingGroup', 'Year',],var_name='BootNumber', value_name= dimension_to_plot)
    
    diseases.to_csv('temp' + str(dimension_to_plot) + str(yr1) + '.csv')
    
################ CSV with Medicalization Scores for Each Diseases from Each Model in Each Time Period

for yr1 in [1980, 1983, 1986, 1989, 1992, 1995, 1998, 2001, 2004, 2007, 2010, 2013, 2016]:
    diseases= pd.read_csv('C:/Users/arsen/Dropbox/R01DiseaseStigma/Disease_list_5.12.20_uncorrupted.csv')
    diseases = diseases[ diseases['Plot'] == 'Yes' ]
    diseases = diseases.drop_duplicates(subset=['Reconciled_Name'])
    diseases = diseases[['PlottingGroup', 'Reconciled_Name']]
    yr3=yr1 + 2
    for bootnum in list(range(0,25)):
        currentmodel1= KeyedVectors.load('C:/Users/arsen/Dropbox/R01DiseaseStigma/LexisNexisNews_Data_Modeling/BootstrappedModels/' + str(yr1) + '_' + str(yr3) + '/CBOW_300d__win10_min50_iter3_'+ str(yr1)+ '_' + str(yr3) + "_boot" + str(bootnum)) #load in desired model

        #adding in three post-hoc word-vectors. This is TEMPORARY and not changing the underlying model. 
        epilepsy_folded = fold_word('epilepsy', 'epileptic', currentmodel1)
        drug_addiction_folded = fold_word('drug_addiction', 'drug_addict', currentmodel1)
        obesity_folded = fold_word('obesity', 'obese', currentmodel1)
        
        currentmodel1.wv.add('epilepsy_folded', epilepsy_folded)
        currentmodel1.wv['drug_addiction_folded'] = drug_addiction_folded
        currentmodel1.wv['obesity_folded'] = obesity_folded
        
        #need to also update the count since this is used to exclude certain diseases
        currentmodel1.wv.vocab['epilepsy_folded'].count = currentmodel1.wv.vocab['epileptic'].count + currentmodel1.wv.vocab['epilepsy'].count
        currentmodel1.wv.vocab['drug_addiction_folded'].count = currentmodel1.wv.vocab['drug_addict'].count + currentmodel1.wv.vocab['drug_addiction'].count
        currentmodel1.wv.vocab['obesity_folded'].count = currentmodel1.wv.vocab['obese'].count + currentmodel1.wv.vocab['obesity'].count
                
        #Medicalization (Cluster)
        medwords_curr= build_lexicon_stigma.dimension_lexicon( currentmodel1, medwords) #cluster
        med = dimension_stigma.dimension(medwords_curr,'cluster')
        allwordssims_med= med.cos_sim(list(currentmodel1.wv.vocab),returnNAs=False)
        diseases['med_score_stdized'+  '_' +str(bootnum)]= diseases['Reconciled_Name'].apply(lambda x: ((med.cos_sim([str(x).lower()], returnNAs=True)[0])- np.mean(allwordssims_med))/np.std(allwordssims_med))
    diseases['Year']= [str(yr1)] * len(diseases)
    dimension_to_plot = 'med'
    diseases = pd.melt(diseases[['Reconciled_Name', 'PlottingGroup', 'Year',  str(dimension_to_plot + '_score_stdized_0'), str(dimension_to_plot + '_score_stdized_1'), str(dimension_to_plot + '_score_stdized_2'), str(dimension_to_plot + '_score_stdized_3'),
                                 str(dimension_to_plot + '_score_stdized_4'), str(dimension_to_plot + '_score_stdized_5'), str(dimension_to_plot + '_score_stdized_6'), str(dimension_to_plot + '_score_stdized_7'), str(dimension_to_plot + '_score_stdized_8'), str(dimension_to_plot + '_score_stdized_9'), 
                                 str(dimension_to_plot + '_score_stdized_10'), str(dimension_to_plot + '_score_stdized_11'), str(dimension_to_plot + '_score_stdized_12'), str(dimension_to_plot + '_score_stdized_13'), str(dimension_to_plot + '_score_stdized_14'), 
                                  str(dimension_to_plot + '_score_stdized_15'), str(dimension_to_plot + '_score_stdized_16'), str(dimension_to_plot + '_score_stdized_17'), str(dimension_to_plot + '_score_stdized_18'), 
                                 str(dimension_to_plot + '_score_stdized_19'), str(dimension_to_plot + '_score_stdized_20'), str(dimension_to_plot + '_score_stdized_21'), str(dimension_to_plot + '_score_stdized_22'), str(dimension_to_plot + '_score_stdized_23'), 
                                 str(dimension_to_plot + '_score_stdized_24') ]],
                    id_vars=['Reconciled_Name', 'PlottingGroup', 'Year',],var_name='BootNumber', value_name= dimension_to_plot)
    
    diseases.to_csv('temp' + str(dimension_to_plot) + str(yr1) + '.csv')

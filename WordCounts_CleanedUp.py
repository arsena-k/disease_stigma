# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:44:54 2021

@author: arsen
"""

import pandas as pd
import os
from gensim.models import Word2Vec
from sklearn import normalize

os.chdir('C:/Users/arsen/Dropbox/R01DiseaseStigma/Analyses/') 


# STEP ONE: COMPUTE WORD COUNTS IN EACH BOOTSTRAPPED MODEL

def fold_word(target, second, wvmodel): #following the convex combination example on wiki
    weight_target = wvmodel.wv.vocab[target].count / (wvmodel.wv.vocab[target].count  + wvmodel.wv.vocab[second].count) 
    weight_second = wvmodel.wv.vocab[second].count / (wvmodel.wv.vocab[target].count  + wvmodel.wv.vocab[second].count) 
    weighted_wv= (weight_target* normalize(wvmodel.wv[target].reshape(1,-1))  ) + (weight_second* normalize(wvmodel.wv[second].reshape(1,-1)) ) 
    return ( normalize(weighted_wv) )
 

for yr1 in [1980, 1983, 1986, 1989, 1992, 1995, 1998, 2001, 2004, 2007, 2010, 2013, 2016]:
    diseases= pd.read_csv('C:/Users/arsen/Dropbox/R01DiseaseStigma/Disease_list_5.12.20_uncorrupted.csv')
    diseases = diseases[ diseases['Plot'] == 'Yes' ]
    diseases = diseases.drop_duplicates(subset=['Reconciled_Name'])
    diseases = diseases[['PlottingGroup', 'Reconciled_Name']]
    yr3=yr1 + 2
    for bootnum in [0,1,2,3, 4, 5, 6,7,8,9,10,11,12,13,14, 15, 16, 17 , 18 ,19, 20 ,21, 22 ,23, 24]:
        currentmodel1= Word2Vec.load('C:/Users/arsen/Dropbox/R01DiseaseStigma/LexisNexisNews_Data_Modeling/BootstrappedModels/' + str(yr1) + '_' + str(yr3) + '/CBOW_300d__win10_min50_iter3_'+ str(yr1)+ '_' + str(yr3) + "_boot" + str(bootnum)) #load in desired model

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
        
        allwords_counts= [currentmodel1.wv.vocab[vocabword].count for  vocabword in list(currentmodel1.wv.vocab)]   
        diseases['wordcount'+  '_' +str(bootnum)]= diseases['Reconciled_Name'].apply(lambda x: currentmodel1.wv.vocab[str(x).lower()].count if str(x).lower() in currentmodel1.wv.vocab else 'NA')

    diseases['Year']= [str(yr1)] * len(diseases)
    dimension_to_plot = 'wordcount'
    
    diseases = pd.melt(diseases[['Reconciled_Name', 'PlottingGroup', 'Year',  str(dimension_to_plot + '_0'), str(dimension_to_plot + '_1'), str(dimension_to_plot + '_2'), str(dimension_to_plot + '_3'),
                                 str(dimension_to_plot + '_4'), str(dimension_to_plot + '_5'), str(dimension_to_plot + '_6'), str(dimension_to_plot + '_7'), str(dimension_to_plot + '_8'), str(dimension_to_plot + '_9'), 
                                 str(dimension_to_plot + '_10'), str(dimension_to_plot + '_11'), str(dimension_to_plot + '_12'), str(dimension_to_plot + '_13'), str(dimension_to_plot + '_14'), 
                                  str(dimension_to_plot + '_15'), str(dimension_to_plot + '_16'), str(dimension_to_plot + '_17'), str(dimension_to_plot + '_18'), 
                                 str(dimension_to_plot + '_19'), str(dimension_to_plot + '_20'), str(dimension_to_plot + '_21'), str(dimension_to_plot + '_22'), str(dimension_to_plot + '_23'), 
                                 str(dimension_to_plot + '_24') ]],
                    id_vars=['Reconciled_Name', 'PlottingGroup', 'Year'],var_name='BootNumber', value_name= dimension_to_plot)

    diseases.to_csv('temp' + str(dimension_to_plot) + str(yr1) + '.csv')



# STEP TWO: AGGREGATE BOOTSTRAPS OF WORD COUNTS FOR EACH TIME WINDOW

dim= 'wordcount'

diseases= pd.read_csv('temp' + str(dim) + '1980.csv')#had to save as csv in utf8
diseases1= pd.read_csv('temp' + str(dim) +'1983.csv')#had to save as csv in utf8
diseases= diseases.append(diseases1)
diseases1= pd.read_csv('temp' + str(dim) + '1986.csv')#had to save as csv in utf8
diseases= diseases.append(diseases1)
diseases1= pd.read_csv('temp' + str(dim) + '1989.csv')#had to save as csv in utf8
diseases= diseases.append(diseases1)
diseases1= pd.read_csv('temp' + str(dim) + '1992.csv')#had to save as csv in utf8
diseases= diseases.append(diseases1)
diseases1= pd.read_csv('temp' + str(dim) + '1995.csv')#had to save as csv in utf8
diseases= diseases.append(diseases1)
diseases1= pd.read_csv('temp' + str(dim) + '1998.csv')#had to save as csv in utf8
diseases= diseases.append(diseases1)
diseases1= pd.read_csv('temp'+ str(dim) +'2001.csv')#had to save as csv in utf8
diseases= diseases.append(diseases1)
diseases1= pd.read_csv('temp'+ str(dim) +'2004.csv')#had to save as csv in utf8
diseases= diseases.append(diseases1)
diseases1= pd.read_csv('temp'+ str(dim) +'2007.csv')#had to save as csv in utf8
diseases= diseases.append(diseases1)
diseases1= pd.read_csv('temp'+ str(dim) +'2010.csv')#had to save as csv in utf8
diseases= diseases.append(diseases1)
diseases1= pd.read_csv('temp'+ str(dim) +'2013.csv')#had to save as csv in utf8
diseases= diseases.append(diseases1)
diseases1= pd.read_csv('temp'+ str(dim) +'2016.csv')#had to save as csv in utf8
diseases= diseases.append(diseases1)


# compute rank order to get 92% confidence interval

grouped = diseases.groupby(['Reconciled_Name', 'Year'])
aggregated= grouped.describe(percentiles=[.04, .5, .96]) #Plotting_Group
aggregated = aggregated.reset_index()
aggregated2= aggregated.merge(diseases, how= 'left', on=['Reconciled_Name', 'Year']) #merge back in  
aggregated2 = aggregated2.drop_duplicates(subset=['Reconciled_Name', 'Year'])
aggregated2 = aggregated2.drop(columns=aggregated2.columns[[2,3,4,5,6,7,8,9,10, 11, 20, -1]])
aggregated2.columns = ['Reconciled_Name', 'Year', "count",	"mean"	, "std"	 ,"min"	, "CI4%",	"CI50%"	,"CI96%"	,"max",	"PlottingGroup",	"Dimension"]

# save final CSV with word count data
aggregated2.to_csv('C:/Users/arsen/Dropbox/R01DiseaseStigma/RESULTS/LexisNexis_News/'+ str(dim)+ '_NOTstandardized_aggregated_temp_92CI.csv') 















































###########

diseases= pd.read_csv('C:/Users/arsen/Dropbox/R01DiseaseStigma/RESULTS/LexisNexis_News/OurDiseases_W2V_measures.csv')


############

curryear=1995


vocab_articles=[] #this is  a list of sentences, drawn from a sample of articles

#bigram_transformer= Phraser.load("C:/Users/arsen/Dropbox/R01DiseaseStigma/LexisNexisNews_Data_Modeling/bigrammer_" + str(curryear) + "_" + str(curryear+2))        

for i in [curryear, curryear+1, curryear+2]:
    file = open('C:/Users/arsen/Dropbox/R01DiseaseStigma/LexisNexisAPI_DataCollection/RawData_NotSynced_To_Desktop/ContempData_' + str(i) + '/all' + str(i)+ 'bodytexts_regexeddisamb_listofarticles', 'rb') #do earliest of the three years to latest
    tfile_split= pickle.load(file) #this is a list where each item in list is an article
    file.close()  
   
    
    tfile_split= [i.split(' SENTENCEBOUNDARYHERE ') for i in tfile_split] #split the articles into sentences
    for article in tfile_split:
        cleaned_articlesentences_for_bigrammer=[]
        for sentences_list in article:
            sentences= sentences_list.split()  
            cleaned_articlesentences_for_bigrammer.extend(sentences) #just adding the sentences themselves, using extend rather than append

            #cleaned_articlesentences_for_bigrammer.extend(bigram_transformer[sentences]) #just adding the sentences themselves, using extend rather than append
        vocab_articles.extend(cleaned_articlesentences_for_bigrammer)
    



d1= [num for elem in vocab_articles for num in elem] #list of all 

#d1.count('the')

diseases['count_1980']= diseases['Reconciled_Name'].apply(lambda x: d1.count(x))
diseases['count_1983']= diseases['Reconciled_Name'].apply(lambda x: d1.count(x))
diseases['count_1986']= diseases['Reconciled_Name'].apply(lambda x: d1.count(x))
diseases['count_1989']= diseases['Reconciled_Name'].apply(lambda x: d1.count(x))
diseases['count_1992']= diseases['Reconciled_Name'].apply(lambda x: d1.count(x)) #this one doesn't have bigrammer

diseases['count_1995']= diseases['Reconciled_Name'].apply(lambda x: d1.count(x))

diseases['count_1998']= diseases['Reconciled_Name'].apply(lambda x: d1.count(x))
diseases['count_2001']= diseases['Reconciled_Name'].apply(lambda x: d1.count(x))
diseases['count_2007']= diseases['Reconciled_Name'].apply(lambda x: d1.count(x))
diseases['count_2010']= diseases['Reconciled_Name'].apply(lambda x: d1.count(x))
diseases['count_2013']= diseases['Reconciled_Name'].apply(lambda x: d1.count(x))
diseases['count_2016']= diseases['Reconciled_Name'].apply(lambda x: d1.count(x))



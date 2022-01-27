# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 10:22:48 2022

@author: arsen
"""

import pandas as pd
import os
os.chdir('C:/Users/arsen/Dropbox/R01DiseaseStigma/Analyses/') 


# FIRST AGGREGATE BOOTSTRAPS FROM EACH DIMENSION AND TIME PERIOD:
    
dim= 'negpostraits'
dim2= 'negpostraits'

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

negpostraits= diseases
negpostraits['bootno'] = negpostraits['BootNumber'].str.split('_', expand=True)[3] #split on _ to get bootnumber itself

# Next, disgust

dim= 'disgust'
dim2= 'disgust'

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


disgust= diseases
#disgust['partial_stigma_score'] = disgust['disgust']


disgust['bootno'] = disgust['BootNumber'].str.split('_', expand=True)[3] #split on _ to get bootnumber itself



# Next, danger
dim= 'danger'
dim2= 'danger'

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


danger= diseases


danger['bootno'] = danger['BootNumber'].str.split('_', expand=True)[3] #split on _ to get bootnumber itself


# Next, impurity

dim= 'impurity'
dim2= 'impurity'

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


impurity= diseases

impurity['bootno'] = impurity['BootNumber'].str.split('_', expand=True)[3] #split on _ to get bootnumber itself

# SECOND, MERGE ALL THESE DIMENSIONS SCORES TOGETHER, AGGREGATE BOOTSTRAPPED SCORES, AND CLEAN UP

mergeddat = disgust.merge(negpostraits, on= ["Reconciled_Name", "Year", 'bootno'])
mergeddat = mergeddat.merge(danger, on= ["Reconciled_Name", "Year", 'bootno'])
mergeddat = mergeddat.merge(impurity, on= ["Reconciled_Name", "Year", 'bootno'])

mergeddat['stigma_index_mean']= mergeddat[['disgust', 'danger', 'impurity', 'negpostraits']].mean(axis=1)
        

cols = [0, 4, 5, 6, 7,8,9,10, 11,13,14, 15,16,17, 18]
mergeddat.drop(mergeddat.columns[cols],axis=1,inplace=True)   #now just have 1 dataset with stimaindex


grouped = mergeddat.groupby(['Reconciled_Name', 'Year']) #group by to describe with 92% CI in next line
aggregated= grouped.describe(percentiles=[.04, .5, .96]) 
aggregated = aggregated.reset_index()
aggregated2= aggregated.merge(diseases, how= 'left', on=['Reconciled_Name', 'Year']) #merge back in  
aggregated2 = aggregated2.drop_duplicates(subset=['Reconciled_Name', 'Year'])


aggregated2 = aggregated2.drop(columns=aggregated2.columns[[2,3, 12, 14, 15, 16]])
aggregated2.columns = ['Reconciled_Name', 'Year', "count",	"mean"	, "std"	 ,"min"	, "CI4%",	"CI50%"	,"CI96%"	,"max",	"PlottingGroup"]
aggregated2['Dimension'] = 'stigmaindex'

# THIRD, WRITE RESULTS TO CSV

aggregated2.to_csv('C:/Users/arsen/Dropbox/R01DiseaseStigma/RESULTS/LexisNexis_News/stigmaindex_aggregated_temp_92CI.csv') 
    


# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 13:33:24 2021

@author: arsen
"""

import pandas as pd
import os

os.chdir('C:/Users/arsen/Dropbox/R01DiseaseStigma/Analyses/') 


    
dim= 'med' #manually change to desired dimension (disgust, impurity, danger, or negpostraits) or to med for medicalization

# FIRST AGGREGATE BOOTSTRAPS FROM TIME PERIOD:

    
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


# SECOND, AGGREGATE BOOTSTRAPPED SCORES, AND CLEAN UP


grouped = diseases.groupby(['Reconciled_Name', 'Year'])

aggregated= grouped.describe(percentiles=[.04, .5, .96]) #92 percent CI
aggregated = aggregated.reset_index()
aggregated2= aggregated.merge(diseases, how= 'left', on=['Reconciled_Name', 'Year']) #merge back in  
aggregated2 = aggregated2.drop_duplicates(subset=['Reconciled_Name', 'Year'])
aggregated2 = aggregated2.drop(columns=aggregated2.columns[[2,3,4,5,6,7,8,9,10, 11, 20, -2, -1]])

aggregated2.columns = ['Reconciled_Name', 'Year', "count",	"mean"	, "std"	 ,"min"	, "CI4%",	"CI50%"	,"CI96%"	,"max",	"PlottingGroup"]
aggregated2['Dimension']= str(dim)


# THIRD, WRITE RESULTS TO CSV

aggregated2.to_csv('C:/Users/arsen/Dropbox/R01DiseaseStigma/RESULTS/LexisNexis_News/'+ str(dim)+ '_aggregated_temp_92CI.csv') 
    








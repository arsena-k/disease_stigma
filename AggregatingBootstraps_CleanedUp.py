# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 13:33:24 2021

@author: arsen
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir('C:/Users/arsen/Dropbox/R01DiseaseStigma/Analyses/') 


######################################

dim= 'danger' #change this to the dimension of interest

diseases= pd.read_csv('temp' + str(dim) + '1980.csv')
diseases1= pd.read_csv('temp' + str(dim) +'1983.csv')
diseases= diseases.append(diseases1)
diseases1= pd.read_csv('temp' + str(dim) + '1986.csv')
diseases= diseases.append(diseases1)
diseases1= pd.read_csv('temp' + str(dim) + '1989.csv')
diseases= diseases.append(diseases1)
diseases1= pd.read_csv('temp' + str(dim) + '1992.csv')
diseases= diseases.append(diseases1)
diseases1= pd.read_csv('temp' + str(dim) + '1995.csv')
diseases= diseases.append(diseases1)
diseases1= pd.read_csv('temp' + str(dim) + '1998.csv')
diseases= diseases.append(diseases1)
diseases1= pd.read_csv('temp'+ str(dim) +'2001.csv')
diseases= diseases.append(diseases1)
diseases1= pd.read_csv('temp'+ str(dim) +'2004.csv')
diseases= diseases.append(diseases1)
diseases1= pd.read_csv('temp'+ str(dim) +'2007.csv')
diseases= diseases.append(diseases1)
diseases1= pd.read_csv('temp'+ str(dim) +'2010.csv')
diseases= diseases.append(diseases1)
diseases1= pd.read_csv('temp'+ str(dim) +'2013.csv')
diseases= diseases.append(diseases1)
diseases1= pd.read_csv('temp'+ str(dim) +'2016.csv')
diseases= diseases.append(diseases1)


# AGGREGATE AND COMPUTE RANK ORDER OF SCORES TO GET CONFIDENCE INTERVALS FOR THIS DIMENSION


grouped = diseases.groupby(['Reconciled_Name', 'Year'])
aggregated= grouped.describe(percentiles=[.04, .5, .96]) 
aggregated = aggregated.reset_index()
aggregated2= aggregated.merge(diseases, how= 'left', on=['Reconciled_Name', 'Year']) 
aggregated2= aggregated2.drop_duplicates(subset=['Reconciled_Name', 'Year'])


# SAVE FINAL CSV WITH AGGREGATED SCORES FOR THIS DIMENSION
aggregated2.to_csv('C:/Users/arsen/Dropbox/R01DiseaseStigma/RESULTS/LexisNexis_News/'+ str(dim)+ '_aggregated_temp_92CI.csv') 
    



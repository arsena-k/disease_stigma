# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 10:23:36 2022

@author: arsen
"""


import pandas as pd
from pylab import rcParams, xlim
import matplotlib.pyplot as plt
import os

os.chdir('C:/Users/arsen/Dropbox/R01DiseaseStigma/RESULTS/LexisNexis_News') 


dim = 'stigmaindex' #stigma, negpostraits, impurity, danger, disgust
dat= pd.read_csv('C:/Users/arsen/Dropbox/R01DiseaseStigma/RESULTS/LexisNexis_News/' + str(dim) + '_aggregated_temp_92CI.csv')

datanames= list(set(dat['PlottingGroup'].values)) #groups of diseases
#datanames= ['addictions','autoimmune','behavioral','cancers','contested','eating disorders','genetic','infectious','mental','musculoskeletal','neurodevelopmental','neurological','other','stis','visual_auditory']
mycolors=  ['Red', 'Green','Blue', 'Yellow', 'Orange',  'Black',  'Pink', 'Purple', 'Plum',  'DarkRed','Magenta',  'LimeGreen',  'Teal',  'Cyan',  'Goldenrod',
            'DarkGrey', 'DarkOliveGreen', 'Gold', 'Wheat', 'Peru','Azure','DeepPink', 'LightCoral'] #21 custom chosen colors, since max 20 disease for a group, if add diseases then may need to add more colors


######## PLOT STIGMA SCORES:  
    #Each plot is for a specific disease group, with y axis is the score for this dimensions, and x axis is time. 
    #Each disease in the group is plotted separately.



for j in set(dat['PlottingGroup']):
    print(j)
    grouped= dat[dat['PlottingGroup']==j]
    grouped= grouped[grouped['count']>19] #only plot if there are at least 19 bootstrapped models with this term
    fig, ax = plt.subplots(1)    
    for i in range(0, len(set(grouped['Reconciled_Name'].values))):
        disease= list(set(grouped['Reconciled_Name'].values))[i]
        #if len((grouped[grouped['Reconciled_Name']==str(disease)]['Year']))>9:     #only plot if there are at least 10 total time points/ models            
        #ax.plot(grouped[grouped['Reconciled_Name']==str(disease)]['Year'], grouped[grouped['Reconciled_Name']==str(disease)]['mean'] , lw=2, label=str(disease), color=mycolors[i])
        ax.plot(grouped[grouped['Reconciled_Name']==str(disease)]['Year'], grouped[grouped['Reconciled_Name']==str(disease)]['CI50%'] , lw=2, label=str(disease), color=mycolors[i])
        ax.fill_between(grouped[grouped['Reconciled_Name']==str(disease)]['Year'], grouped[grouped['Reconciled_Name']==str(disease)]['CI4%'], grouped[grouped['Reconciled_Name']==str(disease)]['CI96%'], facecolor=mycolors[i], alpha=0.5)
        
    ax.set_title(str(dim) + r' of diseases in plotting proup: ' + str(j))
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', ncol=1)
    plt.savefig('C:/Users/arsen/Dropbox/R01DiseaseStigma/RESULTS/LexisNexis_News/' + str(j) + '_' + str(dim) + '_Diseases_92CI_median.jpg',  bbox_inches='tight')    





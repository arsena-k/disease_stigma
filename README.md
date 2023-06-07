# disease_stigma

**This code repository accompanies the paper "[The Stigma of Diseases: Unequal Burden, Uneven Decline](https://osf.io/preprints/socarxiv/7nm9x/)" by Rachel Kahn Best and Alina Arseniev-Koehler.** 

Preprint of paper is available here: https://osf.io/preprints/socarxiv/7nm9x/. Details and context for this code are described in the paper and appendices. 

**Paper abstract:** Why are some diseases more stigmatized than others? And, has disease stigma declined over time? Answers to these questions have been hampered by a lack of comparable data. Using word embedding methods, we analyze 4.7 million news articles to create new measures of stigma for 106 health conditions from 1980-2018. Using mixed effects regressions, we find that behavioral health conditions and preventable diseases attract the strongest connotations of immorality and negative personality traits. Meanwhile, infectious diseases are marked by disgust. These results lend new empirical support to theories that norm enforcement and contagion avoidance drive disease stigma. Challenging existing theories, we find no evidence for a link between medicalization and stigma, and inconclusive evidence on the relationship between advocacy and stigma. Finally, we find that stigma has declined dramatically over time, but only for chronic physical illnesses. In the past four decades, stigma has transformed from a sea of negative connotations surrounding most diseases to two primary conduits of meaning: infectious diseases spark disgust, and behavioral health conditions cue negative stereotypes. These results show that cultural meanings are especially durable when they are anchored by interests, and that cultural changes intertwine in ways that only become visible through large-scale research.



**Final_Search_SymptomsDiseasesList.txt**
* List of search terms used for symptoms and diseases, we used this search term list to collect news articles via the Lexis Nexis API. 
* Note: We do not include code to collect raw news data using the Lexis Nexis API. Our code for this is lightly adapted from code provided to us by the University of Michigan to use the LexisNexis API and uses private API keys. We cannot redstribute the raw data collected from Lexis Nexis API.

**TrainingPhraser_CleanedUp.py**
* Train a phraser on text data from a given time window. One phraser trained per time window. 

**TrainingW2V_Booted_CleanedUp.py**
* Train word2vec models on bootstrapped text data. Use phrasers for each time window trained with the code "TrainingPhraser_CleanedUp.py"

**Validating_OverallW2VModels_CleanedUp.py**
* Validate an overall word2vec model on the WordSim-353 Test
* Validate an overall word2vec model on the Google Analogy Test
* Requires: questions_words_pasted.txt

**Validating_Dimensions in Bootstraps_CleanedUp.py**
* Cross-validation for each of 4 dimensions
* Cosine similarities between all the 4 dimensions
* Most and least similar words to each of 4 dimensions
* Requires: build_lexicon_stigma.py and dimension_stigma.py
* Note: We do not include code or data for comparing our dimensions to human-rated data collected by Pachankis et. al; we cannot distribute data from Pachankis et al.  

**WriteStigmaScores_CleanedUp.py**
* Compute each of 4 stigma scores and medicalization score for each disease, in each model, in each time period. Write results to CSVs (one CSV per dimension, per time window).
* Requires: build_lexicon_stigma.py and dimension_stigma.py, updated_personality_trait_list.csv, Stigma_WordLists.csv, and Disease_list_5.12.20_uncorrupted.csv.

**AggregatingStigmaScores_StigmaIndex_CleanedUp.py**
* Aggregate bootstrapped scores for the time windows and 4 dimensions to get a mean and 92% confidence interval for each disease's mean loading across the 4 dimensions (i.e., stigma score) in each time window. Write results to a single CSV (this CSV is also included in this repository: stigmaindex_aggregated_temp_92CI.csv). 

**AggregatingBootstraps_CleanedUp.py**
* Aggregate bootstrapped scores for time windows to get a mean and 92% confidence interval for each disease's loading on a dimension in a given time window. Write results to a CSV (one CSV per dimension). 

**WordCounts.py**
* Compute the number of mentions for each disease, in each model, in each time period.  Get a mean and 92% confidence interval for each disease's number of mentions in a given time window. Write results to a CSV. 
* Requires: Disease_list_5.12.20_uncorrupted.csv

**PlottingBootstrapped_CleanedUp.py**
* Visualize stigma scores of diseases, by disease group, across time. (Requires stigmaindex_aggregated_temp_92CI.csv).

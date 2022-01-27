# disease_stigma

**This code repository accompanies the paper "Stigma's Uneven Decline" by Rachel Kahn Best and Alina Arseniev-Koehler.** 

Preprint of paper is available here: https://osf.io/preprints/socarxiv/7nm9x/. Details and context for this code are described in the paper and appendices. 

**Paper abstract:** Has disease stigma declined? Our ability to answer this question has been hampered by the lack of comparable data across diseases and over time. Using word embeddings, we analyze 4.5 million news articles to create new measures of stigma for 107 health conditions. We find that in the 1980s, most diseases were marked by strong connotations of disgust, danger, impurity, and negative personality traits. Since then, stigma has declined dramatically for most physical illnesses; cancers, neurological conditions, genetic diseases, and many other conditions have shed most of their negative connotations. But this decline was uneven: mental illnesses, eating disorders, and addictions saw no stigma declines, and stigma declined more slowly for infectious diseases than for chronic conditions. Using multivariate regression, we find that patients’ activism explains some but not all of the variation in stigma. Stigma has transformed from a sea of negative connotations surrounding most diseases to a narrower set of judgments targeting conditions where the primary symptoms are aberrant behaviors.



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

**WordCounts.py (TO DO)**
* Compute the number of mentions for each disease, in each model, in each time period. 
* Requires: Disease_list_5.12.20_uncorrupted.csv

**PlottingDiseases.ipynp (TO DO)**
* Visualize stigma scores of diseases across time

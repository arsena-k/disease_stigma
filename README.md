# disease_stigma
Code repository in progress to accompany the paper "Stigma's Uneven Decline" by Rachel Kahn Best and Alina Arseniev-Koehler. Preprint of paper is available here: https://osf.io/preprints/socarxiv/7nm9x/. Details and context are described in the paper and appendices.  

TrainingPhraser_CleanedUp.py
* Train a phraser on text data from a given time window. One phraser trained per time window. 

TrainingW2V_Booted_CleanedUp.py
* Train word2vec models on bootstrapped data. Use phrasers for each time window trained with the code "TrainingPhraser_CleanedUp.py"

ValidatingStigmaScores.py
* Cross-validation for dimensions
* Cosine similarities between dimensions
* Most and least similar words to each dimension

WriteStigmaScores_CleanedUp.py
* Compute each of 4 stigma scores and medicalization score for each disease, in each model, in each time period. Write results to CSVs.
* Uses functions in:  build_lexicon_stigma.py and dimension_stigma.py

AggregatingStigmaScores_CleanedUp.py (TO DO)
* Aggregate bootstrapped scores for time windows into a mean and confidence interval 

WordCounts.py (TO DO)
* Compute the number of mentions for each disease, in each model, in each time period. Write results to a CSV.

PlottingDiseases.ipynp (TO DO)
* Visualize stigma scores of diseases across time

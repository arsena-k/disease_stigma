# disease_stigma
Repository in progress to accompany the paper "Stigma's Uneven Decline" by Rachel Best and Alina Arseniev-Koehler.



TrainingPhraser_CleanedUp.py
* Train a phraser on text data from a given time window. One phraser trained per time window. 

TrainingW2V_Booted_CleanedUp.py
* Train word2vec models on bootstrapped data. Use phrasers for each time window trained with the code "TrainingPhraser_CleanedUp.py"

WriteStigmaScores_CleanedUp.py
* Computing each of 4 stigma scores and medicalization score for each diseases, in each model, in each time period. Write results to CSVs.
* Uses functions in:  build_lexicon_stigma.py and dimension_stigma.py

AggregatingStigmaScores_CleanedUp.py (TO DO)
* Aggregates bootstrapped scores for time windows into a mean and confidence interval 

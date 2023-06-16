# Script that loads the sentiment scores from the lexicons and predicts the aggregated 
# sentiment score using min-max normalization

from combo_sentiment import *
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# Extract scores from the lexicon_scores dictionary
afinn_scores = np.asarray(lexicon_scores['AFINN'])
vader_scores = np.asarray(lexicon_scores['VADER'])
sentiwordnet_scores = np.asarray(lexicon_scores['SentiWordNet'])
textblob_scores = np.asarray(lexicon_scores['TextBlob'])
pattern_scores = np.asarray(lexicon_scores['Pattern'])

# Define the minimum and maximum possible scores for each lexicon
afinn_min, afinn_max = -5, 5
vader_min, vader_max = -1, 1
sentiwordnet_min, sentiwordnet_max = -1, 1
textblob_min, textblob_max = -1, 1
pattern_min, pattern_max = -1, 1

# Normalize the scores based on the range of each lexicon
afinn_normalized = (afinn_scores - afinn_min) / (afinn_max - afinn_min)
vader_normalized = (vader_scores - vader_min) / (vader_max - vader_min)
sentiwordnet_normalized = (sentiwordnet_scores - sentiwordnet_min) / (sentiwordnet_max - sentiwordnet_min)
textblob_normalized = (textblob_scores - textblob_min) / (textblob_max - textblob_min)
pattern_normalized = (pattern_scores - pattern_min) / (pattern_max - pattern_min)

# Combine the normalized scores into a single numpy array
normalized_all_scores = np.vstack((afinn_normalized, vader_normalized,
                                   sentiwordnet_normalized, textblob_normalized,
                                   pattern_normalized)).T

# Calculate the aggregated score by taking the average of the normalized scores
aggregated_score = np.mean(normalized_all_scores, axis=1)

print("Aggregated Score:", aggregated_score)
# Machine learning script that loads the sentiment scores from the generated dataset and reshapes them into a 
#feature matrix. The loaded fusion model (trained and saved separately) predicts the aggregated 
#sentiment score using the feature matrix.

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load the sentiment scores from the generated dataset
afinn_scores = load_afinn_scores()
vader_scores = load_vader_scores()
sentiwordnet_scores = load_sentiwordnet_scores()
textblob_scores = load_textblob_scores()
pattern_scores = load_pattern_scores()

# Perform Z-score normalization on the sentiment scores
scaler = StandardScaler()
afinn_scores_normalized = scaler.fit_transform(afinn_scores.reshape(-1, 1))
vader_scores_normalized = scaler.fit_transform(vader_scores.reshape(-1, 1))
sentiwordnet_scores_normalized = scaler.fit_transform(sentiwordnet_scores.reshape(-1, 1))
textblob_scores_normalized = scaler.fit_transform(textblob_scores.reshape(-1, 1))
pattern_scores_normalized = scaler.fit_transform(pattern_scores.reshape(-1, 1))

# Reshape the normalized sentiment scores into a feature matrix for the fusion model
X = np.concatenate((afinn_scores_normalized, vader_scores_normalized, sentiwordnet_scores_normalized,
                    textblob_scores_normalized, pattern_scores_normalized), axis=1)

# Load the trained fusion model (Linear Regression model in this example)
model = LinearRegression()
model.load_model("fusion_model.pkl")  # Load the trained model from a saved file

# Predict the aggregated sentiment score using the fusion model
aggregated_score = model.predict(X)

print("Aggregated Score:", aggregated_score)

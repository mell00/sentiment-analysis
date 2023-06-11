# Machine learning script that loads the sentiment scores from the generated dataset and reshapes them into a 
#feature matrix. The loaded fusion model (trained and saved separately) predicts the aggregated 
#sentiment score using the feature matrix.

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Prepare the dataset
# Assuming you have a dataset with individual lexicon scores and sentiment labels
# X represents the feature matrix (lexicon scores), y represents the target variable (sentiment labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Aggregated score calculation for a new text input
# Assuming you have the lexicon scores for the new text stored in a variable called 'new_lexicon_scores'
new_lexicon_scores = [0.2, 0.4, -0.1, 0.5]  # Example scores, replace with actual lexicon scores

# Reshape the lexicon scores to match the shape expected by the model
new_lexicon_scores = np.array(new_lexicon_scores).reshape(1, -1)

# Predict the sentiment label or score for the new text
aggregated_score = model.predict(new_lexicon_scores)

# Use the aggregated_score for further analysis or decision-making
print("Aggregated Score:", aggregated_score)

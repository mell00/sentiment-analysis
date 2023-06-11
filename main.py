#test script for sentiment_prob_dist.py

import nltk
from sentiment_prob_dist import *
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('vader_lexicon')

if __name__ == '__main__':
	text = "I'm not happy with the product."
scores, probabilities = analyze_sentiment(text)

print("Sentiment Scores:", scores)
print("Sentiment Probability Distribution:")
for label, prob in probabilities.items():
    print(label + ": " + str(prob))

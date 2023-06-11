import nltk
from sentiment import *
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('vader_lexicon')

if __name__ == '__main__':
	text = "I'm terribly unhappy with the product."
	scores, label = analyze_sentiment(text)

	print("Sentiment Scores:", scores)
	print("Sentiment Label:", label)

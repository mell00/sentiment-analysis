# updated version of the sentiment analysis script that trains using the AFINN lexicon

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from afinn import Afinn

def analyze_sentiment(text):
    # Initialize the sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    afinn = Afinn()

    # Tokenize the text into words
    words = word_tokenize(text)

    # Check for negation words and adjust sentiment scores accordingly
    negation_words = set(['not', 'no', 'n\'t', 'never', 'without'])
    negation_detected = False
    for i in range(len(words)):
        if words[i].lower() in negation_words:
            negation_detected = not negation_detected
        if negation_detected:
            words[i] = 'not_' + words[i]  # Add "not_" prefix to the word

    # Join the modified words back into a sentence
    modified_text = ' '.join(words)

    # Perform sentiment analysis on the modified text using AFINN lexicon
    afinn_score = afinn.score(modified_text)

    # Perform sentiment analysis on the modified text using VADER
    sentiment_scores = sia.polarity_scores(modified_text)
    compound_score = sentiment_scores["compound"]

    # Determine the sentiment label based on the compound score
    sentiment_label = ""
    if compound_score >= 0.05:
        sentiment_label = "Positive"
    elif compound_score <= -0.05:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    return afinn_score, sentiment_scores, sentiment_label

# Example usage
text = "I'm disappointed with the services this business provides."
afinn_score, scores, label = analyze_sentiment(text)

print("AFINN Score:", afinn_score)
print("Sentiment Scores:", scores)
print("Sentiment Label:", label)

# updated version of the sentiment analysis script that trains using the AFINN, VADER, SentiWordNet, TextBlob, NRCLex, and Pattern lexicons
# and produces an aggregated score to be used in further analyses

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn
from afinn import Afinn
from textblob import TextBlob
from nrclex import NRCLex
from pattern.en import sentiment as pattern_sentiment

def analyze_sentiment(text):
    # Initialize the sentiment analyzers
    sia = SentimentIntensityAnalyzer()
    afinn = Afinn()

    # Perform sentiment analysis using AFINN lexicon
    afinn_score = afinn.score(text)

    # Perform sentiment analysis using VADER
    sentiment_scores = sia.polarity_scores(text)

    # Perform sentiment analysis using SentiWordNet
    sentiwordnet_scores = 0.0
    tokens = nltk.word_tokenize(text)
    for token in tokens:
        synsets = list(swn.senti_synsets(token))
        if synsets:
            sentiwordnet_scores += synsets[0].pos_score() - synsets[0].neg_score()

    # Perform sentiment analysis using TextBlob
    blob = TextBlob(text)
    textblob_polarity = blob.sentiment.polarity

    # Perform sentiment analysis using NRCLex
    nrc = NRCLex(text)
    emotion_scores = nrc.affect_frequencies

    # Perform sentiment analysis using Pattern
    pattern_score = pattern_sentiment(text)[0]

    return afinn_score, sentiment_scores, sentiwordnet_scores, textblob_polarity, emotion_scores, pattern_score

# Example usage
text = "I'm not happy with the product."
afinn_score, sentiment_scores, sentiwordnet_score, textblob_polarity, emotion_scores, pattern_score = analyze_sentiment(text)

# Aggregate the scores or perform comparisons as desired
aggregate_score = (afinn_score + sentiment_scores['compound'] + sentiwordnet_score + textblob_polarity + sum(emotion_scores.values()) + pattern_score) / 6
print("Aggregate Score:", aggregate_score)

print("AFINN Score:", afinn_score)
print("Sentiment Scores (VADER):", sentiment_scores)
print("SentiWordNet Score:", sentiwordnet_score)
print("TextBlob Polarity:", textblob_polarity)
print("Emotion Scores (NRCLex):", emotion_scores)
print("Pattern Score:", pattern_score)

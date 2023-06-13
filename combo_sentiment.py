# updated version of the sentiment analysis script that trains using the AFINN, VADER, SentiWordNet, TextBlob, and Pattern lexicons
# and produces an aggregated score to be used in further analyses

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn
from afinn import Afinn
from textblob import TextBlob
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

    # Perform sentiment analysis using Pattern
    pattern_score = pattern_sentiment(text)[0]

    return afinn_score, sentiment_scores, sentiwordnet_scores, textblob_polarity, pattern_score
    #return {
       # "AFINN": afinn_score,
        #"VADER": sentiment_scores["compound"],
        #"SentiWordNet": sentiwordnet_scores,
        #"TextBlob": textblob_polarity,
        #"Pattern": pattern_score
    #}

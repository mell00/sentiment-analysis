# updated version of the sentiment analysis script that trains using the AFINN, VADER, SentiWordNet, TextBlob, and Pattern lexicons
# and loops through all lines of a paragraph
# (does not produce an aggregated score anymore; see compute_aggregated_score.py)

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

    return create_dictionary(afinn_score,sentiment_scores["compound"],sentiwordnet_scores,textblob_polarity,pattern_score)

def create_dictionary(a,b,c,d,e):
   lexicon_scores = dict();
   lexicon_scores['AFINN'] = a
   lexicon_scores['VADER'] = b
   lexicon_scores['SentiWordNet'] = c
   lexicon_scores['TextBlob'] = d
   lexicon_scores['Pattern'] = e
   return lexicon_scores

def analyze_paragraph(paragraph):
    lines = paragraph.split('\n')
    lexicon_scores_dict = dict()
    for i, line in enumerate(lines):
        lexicon_scores = analyze_sentiment(line)
        lexicon_scores_dict[i] = lexicon_scores
    return lexicon_scores_dict

# Test run
paragraph = "I love this product.\nIt's amazing!\nI hate it though."
lexicon_scores_dict = analyze_paragraph(paragraph)
print('The lexicon score dictionaries for each line are:', lexicon_scores_dict)

# updated version of the sentiment analysis script that trains using the AFINN, VADER, SentiWordNet, TextBlob, EmoLex, and MPQA Subjectivity lexicons

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn
from afinn import Afinn
from textblob import TextBlob
from emotion.EmoLex import EmoLex
from mpqa.MpqaSubjectivityLexicon import MpqaSubjectivityLexicon

def analyze_sentiment(text):
    # Initialize the sentiment analyzers
    sia = SentimentIntensityAnalyzer()
    afinn = Afinn()
    emo_lex = EmoLex()
    mpqa_lex = MpqaSubjectivityLexicon()

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

    # Perform sentiment analysis using EmoLex
    emolex_scores = emo_lex.compute_emotion_scores(text)

    # Perform sentiment analysis using MPQA Subjectivity Lexicon
    mpqa_scores = mpqa_lex.get_scores(text)

    return afinn_score, sentiment_scores, sentiwordnet_scores, textblob_polarity, emolex_scores, mpqa_scores

# Example usage
text = "I'm not happy with the product."
afinn_score, sentiment_scores, sentiwordnet_score, textblob_polarity, emolex_scores, mpqa_scores = analyze_sentiment(text)

print("AFINN Score:", afinn_score)
print("Sentiment Scores (VADER):", sentiment_scores)
print("SentiWordNet Score:", sentiwordnet_score)
print("TextBlob Polarity:", textblob_polarity)
print("EmoLex Scores:", emolex_scores)
print("MPQA Scores:", mpqa_scores)

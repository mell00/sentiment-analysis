# updated version of the sentiment analysis script that estimates the likelihood of the text belonging to each sentiment category

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

def analyze_sentiment(text):
    # Initialize the sentiment analyzer
    sia = SentimentIntensityAnalyzer()

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

    # Perform sentiment analysis on the modified text
    sentiment_scores = sia.polarity_scores(modified_text)

    # Calculate the total score to normalize the probabilities
    total_score = sum(sentiment_scores.values())

    # Calculate the probability distribution
    sentiment_probs = {label: score / total_score for label, score in sentiment_scores.items()}

    return sentiment_scores, sentiment_probs

# Example usage
text = "I'm not happy with the product."
scores, probabilities = analyze_sentiment(text)

print("Sentiment Scores:", scores)
print("Sentiment Probability Distribution:")
for label, prob in probabilities.items():
    print(label + ": " + str(prob))


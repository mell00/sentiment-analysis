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

    # Determine the sentiment label based on the compound score
    sentiment_label = ""
    if sentiment_scores["compound"] >= 0.05:
        sentiment_label = "Positive"
    elif sentiment_scores["compound"] <= -0.05:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    return sentiment_scores, sentiment_label

# Example usage
text = "I'm not happy with the product."
scores, label = analyze_sentiment(text)

print("Sentiment Scores:", scores)
print("Sentiment Label:", label)

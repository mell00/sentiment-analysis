# Example usage
text = "I'm not happy with the product."
afinn_score, sentiment_scores, sentiwordnet_scores, textblob_polarity, pattern_score = analyze_sentiment(text)

# Aggregate the scores or perform comparisons as desired
aggregate_score = (afinn_score + sentiment_scores['compound'] + sentiwordnet_scores + textblob_polarity + pattern_score) / 5
print("Aggregate Score:", aggregate_score)

print("AFINN Score:", afinn_score)
print("Sentiment Scores (VADER):", sentiment_scores)
print("SentiWordNet Score:", sentiwordnet_scores)
print("TextBlob Polarity:", textblob_polarity)
print("Pattern Score:", pattern_score)
#R script that plots a probability distribution of the sentiment scores. To be edited

library(ggplot2)

# Assuming you have a vector of aggregated scores for each line of text
aggregated_scores <- c(0.2, -0.1, 0.3, -0.05, 0.15, 0.0, -0.2, 0.1)

# Define the threshold for sentiment classification
threshold <- 0.1

# Classify sentiments based on the threshold
sentiments <- ifelse(aggregated_scores >= threshold, 'positive',
                     ifelse(aggregated_scores <= -threshold, 'negative', 'neutral'))

# Calculate probabilities
total_count <- length(sentiments)
positive_prob <- sum(sentiments == 'positive') / total_count
neutral_prob <- sum(sentiments == 'neutral') / total_count
negative_prob <- sum(sentiments == 'negative') / total_count

# Create a data frame for plotting
data <- data.frame(Sentiment = c('Positive', 'Neutral', 'Negative'),
                   Probability = c(positive_prob, neutral_prob, negative_prob))

# Plotting the probability distribution
ggplot(data, aes(x = Sentiment, y = Probability, fill = Sentiment)) +
  geom_bar(stat = 'identity') +
  xlab('Sentiment') +
  ylab('Probability') +
  ggtitle('Sentiment Probability Distribution')

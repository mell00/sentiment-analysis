# Load the reticulate package
library(reticulate)

# Import the Python dictionary
py <- import("combo_sentiment")

# Access the Python dictionary in R
lexicon_data <- py$my_dict

# Print the contents of the dictionary
print(lexicon_data)

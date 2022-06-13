# Tweet sentiment Detector


## INTRODUCTION
This is a web based tweet classification system deployed to a remote server. The classifer used is a Naive Bayes classifier trained on a corpus. The documents in the corpus were processed and vectorized using tf-idf vectorizer. A flask app containing a pickled form of the trained model was then deployed to the web for inference.


## INPUT
A tweet text.

## OUTPUT
positive or negative sentiment.

# METHOD
The text is preprocessed and vectorized using the tf-idf vectorizer and then passed to the Naive Bayes classifier which return a probability prediction. If the probability is less than 50%, it carries a negative sentiment and if greater than 50% it carries a positive sentiment

# METRIC
Accuracy - 75%
F1 score, Recall, Precision

For more info on the code, model architecture and explanations, read the .ipynb file attached to this repository.

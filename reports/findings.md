# Findings

## Findings from the Visualizations

The visualizations show that the most prominent words in the data set have terms that are significantly connected to credible and scam messages. This reaffirms the determination of the discriminative linguistic patterns in the text data.

## Findings from the Baseline Model

- **F1 Score:** 0.9825 – The model performs extremely well, especially for a traditional logistic regression model.

- **Confusion Matrix Values:**
- True Positives: 128
- True Negatives: 241
- False Positives: 7
- False Negatives: 1

These values show that the model performs very well at classifying most of the messages. The very low false negative rate shows great recall, suggesting the model is extremely good at picking out scam messages.

## Findings from the Transformer Model

Training loss and test loss got lower after each epoch.

- Training loss: 0.33385
- Test loss: 0.03587

The training loss is significantly lower than the test loss. This could be due to the low amount of data used in training the model and the limited number of epochs (5).

F1 score: 0.926622. This suggests that the transformer model was very good at classifying the test set.

- **Confusion Matrix Values:**
- True Positives: 72
- True Negatives: 21
- False Positives: 7
- False Negatives: 0

## General findings

The base model appears to have performed slightly better than the transformer model. However, this may not be true if a larger dataset was used, as transformers are better suited to capture context and logistic regression is limited to only data that is linearly separable, which may not be true for real-world scam messages.

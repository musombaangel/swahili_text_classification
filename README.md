# Swahili text Classification
This is a project that classifies swahili texts as scam or trustworthy. Two models were used:
1. A baseline logistic regression model
2. A transformer model (XML-R)

# License citation
The dataset used in this project is sourced from Kaggle:
[Swahili SMS Detection Dataset]-[Author name: Henry Dioniz]

The dataset can be found here: https://www.kaggle.com/datasets/henrydioniz/swahili-sms-detection-dataset

# Methodology
1. Baseline model
   - Preprocessing
   - Tokenization using TF-IDF vectorizer
   - Model used was logistic regression from sklearn
   - Evaluation using F1 score and a confusion matrix
    
2. Transformer model
   - Preprocessing using AFRO-XMLR tokenizer for tokenization
   - Model used was Fine-tuned Afro-XLMR

# Results
- The baseline Model achieved an F1 Score of 0.9825
- Results from the transformer model to be updated

# Installation
1. Clone the repository
2. Ensure all requirements and install dependencies:
4. Run the notebooks

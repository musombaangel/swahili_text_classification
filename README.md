# Swahili text Classification
Classification of Swahili messages as scam or trustworthy.<br>
Two models were used in this project:
1. A baseline logistic regression model
2. A transformer model (XML-R)

## File structure
|-- data/<br>
|   |-- bongo_scam.csv<br>
|   |-- bongo_scam_cleaned.csv<br>
|<br>
|-- extra_report/<br>
|   |-- extra_report.py<br>
|   |-- predictions.csv<br>
|<br>
|-- notebooks/<br>
|   |-- baseline_model.py<br>
|   |-- preprocessing.py<br>
|   |-- transformer_model.py<br>
|<br>
|-- reports/<br>
|   |-- findings.md<br>
|   |-- visualizations.ipynb<br>
|<br>
|-- requirements.txt<br>


## Dependencies
The model dependencies are listed under the requirements.txt file. <br>All of them can be installed using pip:
pip install -r requirements.txt

# About the data
The data (bongoscam.csv in the data folder) is a dataset with two columns:
- Sms: A column with swahili texts.
- Category: A column with binary values `scam`(scam messages) and `trust` (not spam).

## License citation
The dataset used in this project is sourced from Kaggle:
[Swahili SMS Detection Dataset]-[Author name: Henry Dioniz]

The dataset can be found here: https://www.kaggle.com/datasets/henrydioniz/swahili-sms-detection-dataset

# Methodology

## Preprocessing
The preprocessing.py file handles all the data processing that is necessary. These steps include:<br>
   - Removal of capital letters, numbers and special characters: Capital letters and special characters may interfere with how words are vectorized. Numbers were removed because they are not useful in determining context given that a lot of messages included them, which could mislead the model.<br>
   - Removal of stop words: These are words that appear commonly and may not be as useful in determining context
     Below is the list of stop words used:<br>
     `["akasema","hii","alikuwa","alisema","baada","basi","bila","cha","chini","hadi","hapo","hata","hivyo","hiyo","huku","huo","ili","ilikuwa","juu","kama","karibu","katika","kila","kima","kisha","kubwa","kutoka","kuwa","kwa","kwamba","kwenda","kwenye","la","lakini","mara","mdogo","mimi","mkubwa","mmoja","moja","muda","mwenye","na","naye","ndani","ng","ni","nini","nonkungu","pamoja","pia","sana","sasa","sauti","tafadhali","tena","tu","vile","wa","wakati","wake","walikuwa","wao","watu","wengine","wote","ya","yake","yangu","yao","yeye","yule","za","zaidi","zake"]`

The preprocessed data is then saved into the data folder for use in the other models:<br>
```python
df.to_csv('data/bongo_scam_cleaned.csv',index=False)
```

## Baseline model
   - Tokenization is done using the TF-IDF vectorizer: a statistical measure that evaluates the importance of a word in a document relative to a collection of documents (corpus).
   - Model used was logistic regression from sklearn: an effective traditional algorithm for classification tasks. 
   - Evaluation using F1 score and confusion matrix:<br>
**F1 Score:** 0.9825 – The model performs extremely well, especially for a traditional logistic regression model.

### Confusion Matrix Values:
- True Positives: 128
- True Negatives: 241
- False Positives: 7
- False Negatives: 1

    
## Transformer model
   - Preprocessing using AFRO-XMLR tokenizer for tokenization
   - Model used was Fine-tuned Afro-XLMR<br>
     **Choice of Transformer Model**:XLM-RoBERTa by hugging face was used because it is pretrained on 100 different languages, that include African languages like swahili unlike BERT models that are trained on a large corpus of English text, with minimal representation from African languages.
   - The model was trained on the first 500 records with a batch size of 8 and 10 epochs
   - Evaluation using F1 score and confusion matrix<br>
**F1 score**: `0.926622` - This suggests that the model was pretty good at classifying the texts.<br>
### Confusion Matrix Values:
- True Positives: 72
- True Negatives: 21
- False Positives: 7
- False Negatives: 0

Further details regarding the model performance are available in the reports folder. The extra_report folder contains visualizations that further explain the dataset and model outputs.

# Installation
1. Clone the repository
2. Ensure all requirements and install dependencies
4. Run the notebooks, starting with the preprocessing

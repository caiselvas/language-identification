# Language Identification using Character Trigrams

Language identification is a task in human language processing (HLP or NLP). In this project, the use of trigrams is explored in order to create a model capable of solving this problem. Additionally, 3 different smoothing functions have been implemented and tested: Lidstone, Linear Discounting, and Absolute Discounting.

## Dataset
The data used as a training and test corpus comes from *Wortschats Leipzig Corpora*, which contains texts in different languages. Specifically, Spanish, Italian, English, French, Dutch, and German have been used.

The training set consists of 30,000 sentences for each language, while the test set has 10,000.

## Evaluation
The evaluation of the model, in terms of accuracy, is 99.8932%, which translates to only 64 errors.
The confusion matrix is as follows:

![image](https://github.com/caiselvass/Language_Identification/assets/117848447/127b53a6-21ef-46ea-b2e3-50516decb30e)

## Contents
- **Original_langId**: Contains the dataset obtained from *Wortschats Leipzig Corpora*.
- **Preprocessed_langId**: Contains the preprocessed datasets.
- **Weights**: Contains the model parameters (both for the test and the validation).
- **Train.py**: Includes code for the text preprocessing and for the creation of json files.
- **Main.ipynb**: Notebook with the validation and the test of the model. Main part.
- **Report**: Detailed documentation on the decisions taken, justifications, results, and conclusions.
- **Requirements**: python 11.+ , sklearn, matplotlib, seaborn, nltk, (spacy in case of detecting proper names).

---
**References**
- *Wortschats Leipzig Corpora*: [Link to Dataset](https://example.com/leipzig_corpora)
- [HLP Course of GIA (UPC)](https://www.fib.upc.edu/ca/estudis/graus/grau-en-intelligencia-artificial/pla-destudis/assignatures/PLH-GIA)

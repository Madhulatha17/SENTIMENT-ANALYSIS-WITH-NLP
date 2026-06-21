# SENTIMENT-ANALYSIS-WITH-NLP


# PROJECT OVERVIEW

This project performs Sentiment Analysis on customer reviews using Natural Language Processing (NLP) techniques. The objective is to classify customer reviews as Positive or Negative based on their textual content.

The project uses TF-IDF Vectorization for feature extraction and Logistic Regression for sentiment classification. The model is trained on a dataset of customer reviews and evaluated using standard performance metrics.


# OBJECTIVES

- Perform text preprocessing on customer reviews.
- Convert text data into numerical features using TF-IDF.
- Train a Logistic Regression model for sentiment classification.
- Evaluate model performance using accuracy and classification metrics.
- Predict sentiments of new customer reviews.

# FEATURES

- Text Cleaning and Preprocessing
- TF-IDF Feature Extraction
- Logistic Regression Classification
- Sentiment Prediction
- Accuracy Evaluation
- Classification Report Generation
- Confusion Matrix Analysis

# TECHNOLOGIES USED

- Python
- Pandas
- Scikit-Learn
- TF-IDF Vectorizer
- Logistic Regression
- Jupyter Notebook

# DATASET

The dataset contains customer reviews labeled as:

- Positive
- Negative

Example Reviews:

| Review | Sentiment |
|----------|------------|
| This product is amazing | Positive |
| Excellent quality | Positive |
| Very poor service | Negative |
| Worst purchase ever | Negative |

# PROJECT WORKFLOW

1. Load Customer Review Dataset
2. Preprocess Text Data
3. Remove Punctuation and Unwanted Characters
4. Convert Text to TF-IDF Features
5. Split Dataset into Training and Testing Sets
6. Train Logistic Regression Model
7. Predict Sentiments
8. Evaluate Model Performance
9. Test on Custom Reviews

# MODEL USED

## Logistic Regression

Logistic Regression is a supervised machine learning algorithm commonly used for binary classification problems such as sentiment analysis.

Advantages:

- Fast Training
- Easy Implementation
- Good Performance on Text Classification
- Interpretable Results


# EVALUATION METRICS

The model is evaluated using:

- Accuracy Score
- Classification Report
- Precision
- Recall
- F1-Score
- Confusion Matrix

# SAMPLE OUTPUT

Model Accuracy:

0.90

Predicted Sentiment:

Review:
Fantastic product and excellent quality

Output:
Positive

Review:
Very bad experience and poor performance

Output:
Negative

# HOW TO RUN

1. Install Required Libraries

```bash
pip install pandas scikit-learn
```

2. Open Jupyter Notebook

```bash
jupyter notebook
```

3. Open:

```text
Sentiment_Analysis.ipynb
```

4. Run all cells.

# RESULTS

- Successfully performed sentiment analysis.
- Converted text data into numerical features using TF-IDF.
- Classified customer reviews using Logistic Regression.
- Evaluated model performance using multiple metrics.
- Predicted sentiments of new reviews accurately.

# CONCLUSION

This project demonstrates Sentiment Analysis using Natural Language Processing techniques. TF-IDF Vectorization and Logistic Regression were used to classify customer reviews into positive and negative sentiments. The model achieved effective performance and can be extended for larger real-world datasets and advanced NLP applications.

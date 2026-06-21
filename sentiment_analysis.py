import pandas as pd
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = {
    "review": [
        "This product is amazing and works perfectly",
        "I love this item, highly recommended",
        "Excellent quality and fast delivery",
        "Very satisfied with my purchase",
        "The product exceeded my expectations",
        "Terrible experience, waste of money",
        "Very poor quality and bad service",
        "I am disappointed with this product",
        "Worst purchase I have ever made",
        "Not worth the price"
    ],
    "sentiment": [
        "Positive",
        "Positive",
        "Positive",
        "Positive",
        "Positive",
        "Negative",
        "Negative",
        "Negative",
        "Negative",
        "Negative"
    ]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

def preprocess_text(text):

    text = text.lower()

    text = re.sub(r'\d+', '', text)

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    return text

df["clean_review"] = df["review"].apply(preprocess_text)

print("\nPreprocessed Reviews:")
print(df[["review", "clean_review"]])

X = df["clean_review"]
y = df["sentiment"]


vectorizer = TfidfVectorizer()

X_tfidf = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

custom_reviews = [
    "Fantastic product and excellent quality",
    "Very bad experience and poor performance"
]

custom_clean = [
    preprocess_text(review)
    for review in custom_reviews
]

custom_vector = vectorizer.transform(custom_clean)

predictions = model.predict(custom_vector)

print("\nCustom Review Predictions:")

for review, sentiment in zip(custom_reviews, predictions):
    print(f"Review: {review}")
    print(f"Predicted Sentiment: {sentiment}")
    print()

print("""
---------------------------------------
SENTIMENT ANALYSIS WITH NLP
---------------------------------------

1. Customer reviews were collected.

2. Text preprocessing was performed:
   - Lowercase conversion
   - Removal of punctuation
   - Removal of numbers

3. TF-IDF Vectorization converted text into numerical features.

4. Logistic Regression was used as the classification model.

5. The model was trained and evaluated.

6. Accuracy, Classification Report, and Confusion Matrix were generated.

7. The model successfully predicts positive and negative sentiments.

---------------------------------------
PROJECT COMPLETED SUCCESSFULLY
---------------------------------------
""")

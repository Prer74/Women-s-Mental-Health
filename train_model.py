import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression  # Correct import
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset with updated labels (happy/sad)
dataset_path = 'womens_journal.csv'  # Update with your dataset path
balanced_final_dataset = pd.read_csv(dataset_path)

# Split the dataset into features (X) and labels (y)
X = balanced_final_dataset['statements']
y = balanced_final_dataset['status']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.85, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Train a Logistic Regression model
logistic_model = LogisticRegression(random_state=42, max_iter=200)
logistic_model.fit(X_train_tfidf, y_train)

# Save the trained logistic model and the TF-IDF vectorizer
with open('logistic_model.pkl', 'wb') as model_file:
    pickle.dump(logistic_model, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully.")

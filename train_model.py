import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load your dataset
fake_data = pd.read_csv('fake.csv')
true_data = pd.read_csv('true.csv')

# Combine the datasets and create the labels
fake_data['label'] = 0  # Fake news label
true_data['label'] = 1  # True news label

# Concatenate both datasets
data = pd.concat([fake_data, true_data], ignore_index=True)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Use TfidfVectorizer instead of CountVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)

# Fit and transform the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train the model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# Save the model and vectorizer
joblib.dump(model, 'model/model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')

print("Model and vectorizer have been saved successfully.")

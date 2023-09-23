import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load your dataset (replace 'preprocessed_data.csv' with your dataset's file path)
df = pd.read_csv('DataTrain.csv')

# Replace NaN values in the 'text' column with an empty string
df['text'].fillna('', inplace=True)

# Assuming your dataset has 'text' and 'sentiment' columns
X_text = df['text']  # Text data
Y = df['sentiment']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_text, Y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust max_features as needed
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Logistic Regression classifier
classifier = LogisticRegression()
classifier.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive'])

print(f'Accuracy: {accuracy:.2f}')
print(report)

# Make predictions on new data
new_texts = ["I love this product!", "This is terrible...", "It makes me happy."]
new_texts_tfidf = tfidf_vectorizer.transform(new_texts)
new_predictions = classifier.predict(new_texts_tfidf)
print(new_predictions)

model_filename = 'sentiment_classifier.joblib'
joblib.dump(classifier, model_filename)

# Load the saved model
loaded_classifier = joblib.load(model_filename)

# Use the loaded model to make predictions
loaded_predictions = loaded_classifier.predict(new_texts_tfidf)
print(loaded_predictions)

from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib  
# Load your dataset and train your model
df = pd.read_csv('DataTrain.csv')
X_text = df['text']
Y = df['sentiment']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_text, Y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
classifier = LogisticRegression()
classifier.fit(X_train_tfidf, y_train)


# ... (Your previous code for initializing Flask, routes, and preprocessing)

# Load the saved model using joblib
model_filename = 'sentiment_classifier.joblib'
loaded_classifier = joblib.load(model_filename)
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
# Define a route to handle the result page
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['user_input']
        
        # Vectorize the input text (ensure you have the tfidf_vectorizer loaded)
        input_text_tfidf = tfidf_vectorizer.transform([text])
        
        # Make predictions using the loaded model
        prediction = loaded_classifier.predict(input_text_tfidf)[0]

        # Map the numeric prediction back to sentiment labels
        sentiment_labels = ['negative', 'neutral', 'positive']
        #predicted_sentiment = sentiment_labels[int(prediction)]
        predicted_sentiment = prediction
        return render_template('index.html', prediction=predicted_sentiment)

if __name__ == '__main__':
    app.run(debug=True)






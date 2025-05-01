

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')

# Load dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Clean the text data
corpus = []
stemmer = PorterStemmer()
for review in dataset['Review']:
    review = re.sub('[^a-zA-Z]', ' ', review)             # Keep only letters
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in stopwords.words('english') and len(word) > 2]
    corpus.append(' '.join(review))

# TF-IDF with bigrams
tfidf = TfidfVectorizer(max_features=2500, ngram_range=(1, 2))
X = tfidf.fit_transform(corpus).toarray()
y = dataset['Liked'].values

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression + GridSearch for best parameters
params = {
    'C': [0.1, 1.0, 10.0],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [100, 200]
}

grid = GridSearchCV(LogisticRegression(), param_grid=params, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# Predict and evaluate
y_pred = grid.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Model training complete with GridSearch!")
print("🔍 Best Parameters:", grid.best_params_)
print("📊 Accuracy on test data:", round(accuracy * 100, 2), "%")
# === Manual Input for Testing ===
print("\n🔎 You can now test the model with your own review!")

while True:
    user_input = input("Type a food review (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break

    # Clean the input like we did during training
    review = re.sub('[^a-zA-Z]', ' ', user_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in stopwords.words('english') and len(word) > 2]
    cleaned_review = ' '.join(review)

    # Transform using trained TF-IDF vectorizer
    input_features = tfidf.transform([cleaned_review]).toarray()

    # Predict
    prediction = grid.predict(input_features)[0]
    sentiment = "Positive 😀" if prediction == 1 else "Negative 😞"

    print("💬 Sentiment Prediction:", sentiment)


import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Load the training dataset
file_path = "Genre Classification Dataset/train_data.txt"

data = pd.read_csv(
    file_path,
    sep=r"\s*:::\s*",
    engine="python",
    names=["ID", "TITLE", "GENRE", "PLOT"]
)

# 2. Remove rows with missing plots or genres
data = data.dropna(subset=["PLOT", "GENRE"])

# 3. Select input and target
X = data["PLOT"].astype(str)
y = data["GENRE"]

# 4. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 5. Create the TF-IDF tool
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=50000
)

# 6. Learn vocabulary and weights from training plots
X_train_tfidf = tfidf.fit_transform(X_train)

# 7. Transform test plots using the same learned vocabulary
X_test_tfidf = tfidf.transform(X_test)

# 8. Display the results
print("Number of training movies:", len(X_train))
print("Number of testing movies:", len(X_test))
print("Training matrix shape:", X_train_tfidf.shape)
print("Testing matrix shape:", X_test_tfidf.shape)

# 9. Create the machine-learning model
model = LogisticRegression(max_iter=1000)

# 10. Train the model
model.fit(X_train_tfidf, y_train)

print("Model training completed!")
# 11. Predict genres for the testing data
y_pred = model.predict(X_test_tfidf)

# 12. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# 13. Display detailed performance metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 14. Test the model with a new movie plot

new_plot = input("Enter a movie plot: ")

# Convert the new plot into TF-IDF features
new_plot_tfidf = tfidf.transform([new_plot])

# Predict the genre
predicted_genre = model.predict(new_plot_tfidf)

print("Predicted Movie Genre:", predicted_genre[0])

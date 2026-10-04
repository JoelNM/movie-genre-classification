import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

file_path = "Genre Classification Dataset/train_data.txt"

data = pd.read_csv(
    file_path,
    sep=" ::: ",
    engine="python",
    names=["ID", "TITLE", "GENRE", "PLOT"]
)

print("Total rows and columns:", data.shape)

print("\nFirst movie:")
print("ID:", data.loc[0, "ID"])
print("Title:", data.loc[0, "TITLE"])
print("Genre:", data.loc[0, "GENRE"])
print("Plot:", data.loc[0, "PLOT"])
# Separate input and target
X = data["PLOT"]
y = data["GENRE"]

print("Input data:")
print(X.head())

print("\nTarget data:")
print(y.head())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))
# Separate input and target
X = data["PLOT"]
y = data["GENRE"]

print("Input data:")
print(X.head())

print("\nTarget data:")
print(y.head())
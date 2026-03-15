import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------- DATA ----------------

df = pd.read_csv("spam.csv")

print(df.head())
print()

x = df["Message"]
y = df["Category"]


# ---------------- SPLIT ----------------

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ---------------- TF-IDF ----------------

vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(x_train)
X_test_vec = vectorizer.transform(x_test)


# ---------------- MODELS ----------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(kernel="linear", C=1),
    "Naive Bayes": MultinomialNB()
}


best_score = 0
best_model = None
best_name = ""


# ---------------- TRAIN ALL ----------------

for name, model in models.items():

    model.fit(X_train_vec, y_train)

    pred = model.predict(X_test_vec)

    score = accuracy_score(y_test, pred)

    print(name, ":", score)

    if score > best_score:
        best_score = score
        best_model = model
        best_name = name


print("\nBest model:", best_name)
print("Best score:", best_score)


# ---------------- REPORT ----------------

pred = best_model.predict(X_test_vec)

print("\nClassification Report:\n")
print(classification_report(y_test, pred))


# ---------------- TEST WITH 2 MESSAGES ----------------

print("\n--- Testing with 2 messages ---")

messages = [
    "Congratulations! You won a free iPhone. Click now to claim",
    "Hey, are we meeting tomorrow at college?"
]

for msg in messages:
    vec = vectorizer.transform([msg])

    pred = best_model.predict(vec)

    print("\nMessage:", msg)
    print("Prediction:", pred[0])

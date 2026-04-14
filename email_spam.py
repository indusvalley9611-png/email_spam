import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

df = pd.read_csv("spam.csv")

x = df["Message"]
y = df["Category"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(x)

model = SVC(kernel="linear")
model.fit(X, y)

joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

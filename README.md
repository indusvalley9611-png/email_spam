# 📩 Spam Message Classifier

A Machine Learning project that detects whether a message is **spam** or **ham** using multiple classification algorithms.

---

## 🚀 Models Used

* Logistic Regression
* Random Forest
* Decision Tree
* Support Vector Machine (SVM)
* Naive Bayes

The best model is selected based on accuracy.

---

## 📦 Requirements

Install required libraries:

```bash
pip install pandas scikit-learn
```

Recommended Python version:

```
Python 3.10 - 3.12
```

---

## 📂 Dataset Format

Your `spam.csv` must look like this:

```
Category,Message
spam,Win money now
ham,Hello friend
```

Column names must be exactly:

```
Category
Message
```

---

## ▶️ How to Run

Place `spam.csv` in the same folder as the script.

Run:

```bash
python spam_classifier.py
```

---

## 📊 Output

The program will show:

* Accuracy of each model
* Best model
* Classification report
* Prediction for test messages

Example:

```
Message: Congratulations! You won a free iPhone
Prediction: spam

Message: Hey, are we meeting tomorrow?
Prediction: ham
```

---

## 👨‍💻 Author

Machine Learning Practice Project

Prediction: ham

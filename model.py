import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib


# Read original dataset
df = pd.read_csv("D:/IrisProject/data/iris.csv")

X = df.drop("species", axis=1)
y = df["species"]

# split data into train and test sets
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# create an instance of the random forest classifier
clf = RandomForestClassifier(n_estimators=20)

# train the classifier on the training data
clf.fit(X_train, y_train)

# predict on the test set
y_pred = clf.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {accuracy}")  # Accuracy: 0.91

# save the model to disk
joblib.dump(clf, "rf_model.sav")

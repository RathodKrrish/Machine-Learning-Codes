import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Titanic dataset
url = 'titanic.csv'
data = pd.read_csv(url)

# Feature Engineering: Fill missing Age, convert Sex to numbers
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Select features
X = data[['Pclass', 'Sex', 'Age', 'Fare']]
y = data['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
print("Titanic Dataset Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))

# 2)Model Training and Versioning using a Simple Dataset:
# In this assignment, you will work with a small dataset (e.g., the Iris dataset) to build and manage versions of a basic machine learning model.


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

model_v1 = KNeighborsClassifier()
model_v1.fit(X_train, y_train)
y_pred = model_v1.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Tuned Model Accuracy: {accuracy_best}")

print(f"Original Model Accuracy: {accuracy}")
print(f"Tuned Model Accuracy: {accuracy_best}")


import joblib

joblib.dump(model_v1, 'model_v1.pkl')

joblib.dump(best_model, 'model_v2.pkl')

print("Models Saved as model_v1.pkl and model_v2.pkl")





# 3)Saving and Reusing a Machine Learning Model:
# In this assignment, you will train a machine learning model using a simple dataset and learn how to save and reuse the model without retraining.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = KNeighborsClassifier()
model.fit(X_train, y_train)

import joblib
joblib.dump(model, 'model.pkl')

model = joblib.load('model.pkl')
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))





# 4. Creating a Reproducible ML Pipeline using Jupyter and Virtual Environment:
 

from sklearn.datasets import load_wine
import pandas as pd

data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.head()

from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))






# 5. Exploratory Data Analysis (EDA) and Report Generation

import seaborn as sns
import pandas as pd

df = sns.load_dataset("titanic")
print(df.head())

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = sns.load_dataset("titanic")

df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

sns.countplot(x='survived', data=df)
plt.title("Survival Count")
plt.show()
sns.histplot(df['age'], kde=True)
plt.title("Age Distribution")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
df = sns.load_dataset("titanic")

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

sns.histplot(df['fare'], kde=True, bins=30)
plt.title("Fare Distribution")
plt.show()

sns.boxplot(x=df['fare'])
plt.title("Fare Outliers")
plt.show()



from pandas_profiling import ProfileReport

profile = ProfileReport(df, title="Titanic EDA Report", explorative=True)
profile.to_file("titanic_eda_report.html")  







# 6. Visualizing Model Performance
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, accuracy_score

df = sns.load_dataset("titanic")

df['age'] = df['age'].fillna(df['age'].median())  # Assign result back to the column
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])  # Assign result back to the column
df.dropna(subset=['survived', 'pclass', 'age', 'fare'], inplace=True)  # This can remain as is

X = df[['pclass', 'age', 'fare']]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

log_reg = LogisticRegression(max_iter=200)
rf = RandomForestClassifier()

log_reg.fit(X_train, y_train)
rf.fit(X_train, y_train)

log_reg_pred = log_reg.predict(X_test)
rf_pred = rf.predict(X_test)


log_reg_acc = accuracy_score(y_test, log_reg_pred)
rf_acc = accuracy_score(y_test, rf_pred)

print(f"Logistic Regression Accuracy: {log_reg_acc:.4f}")
print(f"Random Forest Accuracy: {rf_acc:.4f}")

log_reg_cm = confusion_matrix(y_test, log_reg_pred)
print(f"Logistic Regression Confusion Matrix:\n{log_reg_cm}")

rf_cm = confusion_matrix(y_test, rf_pred)
print(f"Random Forest Confusion Matrix:\n{rf_cm}")

log_reg_precision, log_reg_recall, _ = precision_recall_curve(y_test, log_reg_pred)
plt.figure(figsize=(10, 6))
plt.plot(log_reg_recall, log_reg_precision, label="Logistic Regression")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Logistic Regression")
plt.legend()
plt.show()

rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_pred)
plt.figure(figsize=(10, 6))
plt.plot(rf_recall, rf_precision, label="Random Forest", color='orange')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Random Forest")
plt.legend()
plt.show()

from sklearn.metrics import ConfusionMatrixDisplay

log_reg_cm = confusion_matrix(y_test, log_reg_pred)
ConfusionMatrixDisplay(log_reg_cm).plot(cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.savefig('results/log_reg_confusion_matrix.png')
plt.show()

rf_cm = confusion_matrix(y_test, rf_pred)
ConfusionMatrixDisplay(rf_cm).plot(cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.savefig('results/rf_confusion_matrix.png')
plt.show()


log_reg_precision, log_reg_recall, _ = precision_recall_curve(y_test, log_reg.predict_proba(X_test)[:, 1])
plt.plot(log_reg_recall, log_reg_precision)
plt.title("Logistic Regression Precision-Recall Curve")
plt.savefig('results/log_reg_precision_recall.png')
plt.show()

rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf.predict_proba(X_test)[:, 1])
plt.plot(rf_recall, rf_precision)
plt.title("Random Forest Precision-Recall Curve")
plt.savefig('results/rf_precision_recall.png')
plt.show()

from sklearn.metrics import accuracy_score

log_reg_acc = accuracy_score(y_test, log_reg_pred)
rf_acc = accuracy_score(y_test, rf_pred)

print("Logistic Regression Accuracy:", round(log_reg_acc, 2))
print("Random Forest Accuracy:", round(rf_acc, 2))




import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import os

train_path = "/Users/admin/Downloads/train_data.csv"
test_path = "/Users/admin/Downloads/test_data.csv"
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

df_test.head(50)

columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked']
df_train = df_train.drop(columns=columns_to_drop)
df_test = df_test.drop(columns=columns_to_drop)
df_train['Sex'] = df_train['Sex'].replace({'male': 0, 'female': 1})
df_test['Sex'] = df_test['Sex'].replace({'male': 0, 'female': 1})

df_train.head(50)

df_test.head(50)

df_train.corr()

survival_rate = df_train['Survived'].mean() * 100

survival_data = pd.DataFrame({'Survival Status': ['Survived', 'Not Survived'],
                              'Percentage': [survival_rate, 100 - survival_rate]})

plt.figure(figsize=(8, 6))
plt.bar(survival_data['Survival Status'], survival_data['Percentage'], color=['green', 'red'])
plt.title('Survival Rate on Titanic')
plt.xlabel('Survival Status')
plt.ylabel('Percentage')
plt.ylim(0, 100)
plt.show()

survival_by_gender = df_train.groupby('Sex')['Survived'].mean() * 100

plt.figure(figsize=(8, 6))
survival_by_gender.plot(kind='bar', color=['blue', 'pink'])
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate (%)')
plt.xticks(rotation=0)
plt.ylim(0, 100)
plt.show()

survived = df_train[df_train['Survived'] == 1]['Fare'].dropna()
not_survived = df_train[df_train['Survived'] == 0]['Fare'].dropna()

plt.figure(figsize=(10, 6))

plt.boxplot([survived, not_survived], labels=['Survived', 'Not Survived'])

plt.title('Survival by Fare')
plt.ylabel('Fare')
plt.grid(True)
plt.show()

survived = df_train[df_train['Survived'] == 1]['Age'].dropna()
not_survived = df_train[df_train['Survived'] == 0]['Age'].dropna()

plt.figure(figsize=(10, 6))

plt.hist(survived, bins=20, alpha=0.5, color='blue', label='Survived')

plt.hist(not_survived, bins=20, alpha=0.5, color='red', label='Not Survived')

plt.title('Survival by Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.show()

y = df_train["Survived"]
X = pd.get_dummies(df_train[["Pclass", "Sex", "SibSp", "Parch", "Fare"]])
X_test = pd.get_dummies(df_test[["Pclass", "Sex", "SibSp", "Parch", "Fare"]])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

correct_predictions = (predictions == df_test["Survived"].values).sum()
total_predictions = len(df_test)
accuracy = (correct_predictions / total_predictions) * 100
print(accuracy)

cm = confusion_matrix(df_test["Survived"].values, predictions)

df_cm = pd.DataFrame(cm, columns=['Not Survived', 'Survived'], index=['Not Survived', 'Survived'])
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'

plt.figure(figsize=(10, 7))
sns.heatmap(df_cm / np.sum(df_cm), fmt='.2%', annot=True, annot_kws={'size': 16})
plt.show()

average_age_male = df_train.loc[df_train['Sex'] == 0, 'Age'].mean()
average_age_female = df_train.loc[df_train['Sex'] == 1, 'Age'].mean()
print(average_age_male)
print(average_age_female)

df_train.loc[df_train['Sex'] == 0, 'Age'] = df_train.loc[df_train['Sex'] == 0, 'Age'].fillna(average_age_male)
df_train.loc[df_train['Sex'] == 1, 'Age'] = df_train.loc[df_train['Sex'] == 1, 'Age'].fillna(average_age_female)
df_test.loc[df_test['Sex'] == 0, 'Age'] = df_test.loc[df_test['Sex'] == 0, 'Age'].fillna(average_age_male)
df_test.loc[df_test['Sex'] == 1, 'Age'] = df_test.loc[df_test['Sex'] == 1, 'Age'].fillna(average_age_female)
df_test['Age'] = df_test['Age'].round()
df_train['Age'] = df_train['Age'].round()

df_train.head(50)

df_test.head(50)

df_train.corr()

y = df_train["Survived"]
X = pd.get_dummies(df_train[["Pclass", "Sex", "SibSp", "Parch", "Age", "Fare"]])
X_test = pd.get_dummies(df_test[["Pclass", "Sex", "SibSp", "Parch", "Age", "Fare"]])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

correct_predictions = (predictions == df_test["Survived"].values).sum()
total_predictions = len(df_test)
accuracy = (correct_predictions / total_predictions) * 100
print(accuracy)

cm = confusion_matrix(df_test["Survived"].values, predictions)

df_cm = pd.DataFrame(cm, columns=['Not Survived', 'Survived'], index=['Not Survived', 'Survived'])
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'

plt.figure(figsize=(10, 7))
sns.heatmap(df_cm / np.sum(df_cm), fmt='.2%', annot=True, annot_kws={'size': 16})
plt.show()

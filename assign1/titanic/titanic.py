import time
import os

# data analysis and wrangling
import pandas as pd
import numpy as np

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.width', 1000)

# reading the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# drop ticket and cabin
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

print('Original data:')
print(train_df.head(20))
print('...')
print(train_df.tail(20))

# sex to ordinal
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

# add missing fare, there is one missing fare
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

# create fare band
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

# convert fare to ordinal values
for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

# create title feature from names
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# normalize titles
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# title to ordinal
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# drop name and passangerid
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

# guess the age for passengers with missing value
guess_ages = np.zeros((2, 3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &
                               (dataset['Pclass'] == j + 1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1),
                        'Age'] = guess_ages[i, j]

    dataset['Age'] = dataset['Age'].astype(int)

# creating age bands
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

# replace age with ordinals
for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

# remove age band feature
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]

# create FamilySize feature from sibsp and parch
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# create 3 categories by family size
for dataset in combine:
    dataset.loc[dataset['FamilySize'] == 1, 'FamilySize'] = 0 # alone
    dataset.loc[(dataset['FamilySize'] > 1) & (dataset['FamilySize'] < 5), 'FamilySize'] = 1
    dataset.loc[(dataset['FamilySize'] > 4), 'FamilySize'] = 2

# drop parch, sibsp and familySize in favor of isAlone
train_df = train_df.drop(['Parch', 'SibSp'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp'], axis=1)
combine = [train_df, test_df]

# most frequent port
freq_port = train_df.Embarked.dropna().mode()[0]

# fill in the missing embarked port
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

# convert embark to numeric
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

print("Final data structure:")
print(train_df.head(20))
print('...')
print(train_df.tail(20))

# training data
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId", axis=1).copy()

# Logistic Regression
print("Training Logistic Regression model...")
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
model_logreg = logreg.predict(X_test)
acc_log = logreg.score(X_train, Y_train) * 100

# calculate coefficients
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

# Support Vector Machines
print("Training SVM model...")
svc = SVC()
svc.fit(X_train, Y_train)
model_svc = svc.predict(X_test)
acc_svc = svc.score(X_train, Y_train) * 100

# K nearest neighbors
print("Training KNN model...")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
model_knn = knn.predict(X_test)
acc_knn = knn.score(X_train, Y_train) * 100

# Gaussian Naive Bayes
print("Training Gaussian Naive Bayes model...")
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
model_gaussian = gaussian.predict(X_test)
acc_gaussian = gaussian.score(X_train, Y_train) * 100

# Perceptron
print("Training Perceptron model...")
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
model_perceptron = perceptron.predict(X_test)
acc_perceptron = perceptron.score(X_train, Y_train) * 100

# Linear SVC
print("Training Linear SVC model...")
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
model_linear_svc = linear_svc.predict(X_test)
acc_linear_svc = linear_svc.score(X_train, Y_train) * 100

# Stochastic Gradient Descent
print("Training Stochastic Gradient Descent model...")
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
model_sgd = sgd.predict(X_test)
acc_sgd = sgd.score(X_train, Y_train) * 100

# Decision Tree
print("Training Decision Tree model...")
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
model_decision_tree = decision_tree.predict(X_test)
acc_decision_tree = decision_tree.score(X_train, Y_train) * 100

# Random Forest
print("Training Random Forest model...")
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
model_random_forest = random_forest.predict(X_test)
acc_random_forest = random_forest.score(X_train, Y_train) * 100

# model evaluation
models = pd.DataFrame({
    'Name': ['Support Vector Machines', 'KNN', 'Logistic Regression',
             'Random Forest', 'Naive Bayes', 'Perceptron',
             'Stochastic Gradient Decent', 'Linear SVC',
             'Decision Tree'],
    'Abbrev': ['SVM', 'KNN', 'LOG_REG', 'RF', 'GNB', 'PERC', 'SGD', 'L_SVC', 'D_TREE'],
    'Score': [acc_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_linear_svc, acc_decision_tree],
    'Model': [model_svc, model_knn, model_logreg,
              model_random_forest, model_gaussian, model_perceptron,
              model_sgd, model_linear_svc, model_decision_tree]})

print(models[['Name', 'Score']].sort_values(by='Score', ascending=False))

selected_model0 = models.sort_values(by='Score', ascending=False).iloc[0]
selected_model1 = models.sort_values(by='Score', ascending=False).iloc[1]
selected_model2 = models.sort_values(by='Score', ascending=False).iloc[2]
print('Selecting model ' + selected_model0['Name'] + ', ' + selected_model1['Name'] + ', ' + selected_model2['Name'])

submission0 = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": selected_model0["Model"]
})
submission1 = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": selected_model1["Model"]
})
submission2 = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": selected_model2["Model"]
})

folder = 'results/' + str(round(time.time()))
os.makedirs(folder)
submission0.to_csv(folder + '/submission'  + '_' + selected_model0['Abbrev'] + '.csv', index=False)
submission1.to_csv(folder + '/submission'  + '_' + selected_model1['Abbrev'] + '.csv', index=False)
submission2.to_csv(folder + '/submission'  + '_' + selected_model2['Abbrev'] + '.csv', index=False)

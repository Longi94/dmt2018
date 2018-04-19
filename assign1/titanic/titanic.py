import time
import os

# data analysis and wrangling
import pandas as pd

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

sex_mapping = {'female': 1, 'male': 0}
port_mapping = {'S': 0, 'C': 1, 'Q': 2}
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}


def main():
    # reading the data
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    combine = [train_df, test_df]

    # drop ticket and cabin
    combine[0] = combine[0].drop(['Ticket'], axis=1)
    combine[1] = combine[1].drop(['Ticket'], axis=1)

    print('Original data:')
    print_preview(combine[0])

    sex_to_ordinal(combine)

    fill_fare(combine[0], combine[1])
    band_fare(combine)

    fill_embarked(combine)
    embarked_to_ordinal(combine)

    transform_cabin(combine)

    create_title(combine)

    create_family_size(combine)

    fill_age(combine)

    create_child(combine)
    create_mother(combine)

    band_age(combine)

    # drop parch, sibsp and familySize in favor of isAlone
    combine[0] = combine[0].drop(['Parch', 'SibSp'], axis=1)
    combine[1] = combine[1].drop(['Parch', 'SibSp'], axis=1)

    print("Final data structure:")
    print_preview(combine[0])

    combine[0].to_csv("processed_train.csv")

    train_models(combine[0], combine[1])


def print_preview(df):
    print(df.head(20))
    print('...')
    print(df.tail(20))


# sex to ordinal
def sex_to_ordinal(combine):
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map(sex_mapping).astype(int)


def fill_fare(train_df, test_df):
    # add missing fare, there is one missing fare with S and 3
    full = pd.concat([train_df, test_df])
    fare_guess = full.loc[(full['Embarked'] == 'S') & (full['Pclass'] == 3), 'Fare'].mean()
    test_df['Fare'] = test_df['Fare'].fillna(fare_guess)


def fill_embarked(combine):
    # fill in the missing embarked port
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna('C')


def embarked_to_ordinal(combine):
    # convert embark to numeric
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)


def band_fare(combine):
    # create fare band
    combine[0]['FareBand'] = pd.qcut(combine[0]['Fare'], 4)

    # convert fare to ordinal values
    for dataset in combine:
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

    combine[0] = combine[0].drop(['FareBand'], axis=1)


def create_title(combine):
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
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

    # drop name and passangerid
    combine[0] = combine[0].drop(['Name', 'PassengerId'], axis=1)
    combine[1] = combine[1].drop(['Name'], axis=1)


def fill_age(combine):
    x_age_train = combine[0].loc[combine[0]["Age"].notnull()].drop("Age", axis=1)
    y_age_train = combine[0].loc[combine[0]["Age"].notnull()]["Age"].astype(int)

    print("Training Decision Tree model for predicting age...")
    print(x_age_train.info())
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(x_age_train, y_age_train)

    for dataset in combine:
        x_age_test = dataset.drop("Age", axis=1).copy()
        model_decision_tree = decision_tree.predict(x_age_test)
        dataset['PredictedAge'] = model_decision_tree
        dataset.loc[dataset['Age'].isnull(), 'Age'] = dataset['PredictedAge']
        dataset['Age'] = dataset['Age'].astype(int)

    combine[0] = combine[0].drop(['PredictedAge'], axis=1)
    combine[1] = combine[1].drop(['PredictedAge'], axis=1)


def band_age(combine):
    # creating age bands
    combine[0]['AgeBand'] = pd.cut(combine[0]['Age'], 5)

    # replace age with ordinals
    for dataset in combine:
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age'] = 4

    # remove age band feature
    combine[0] = combine[0].drop(['AgeBand'], axis=1)


def create_family_size(combine):
    # create FamilySize feature from sibsp and parch
    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[(dataset['SibSp'] + dataset['Parch'] + 1) == 1, 'IsAlone'] = 1


def create_child(combine):
    for dataset in combine:
        dataset['Child'] = 0
        dataset.loc[dataset['Age'] < 18, 'Child'] = 1


def create_mother(combine):
    for dataset in combine:
        dataset['Mother'] = 0
        dataset.loc[(dataset['Child'] == 0) & (dataset['Parch'] > 0) & (dataset['Sex'] == sex_mapping['female']) &
                    (dataset['Title'] != title_mapping['Miss']), 'Mother'] = 1


def transform_cabin(combine):
    for dataset in combine:
        dataset.loc[dataset.Cabin.notnull(), 'Cabin'] = dataset['Cabin'].astype(str).str[0].apply(lambda x: ord(x)) \
                                                        - ord('A') + 1
        dataset.loc[dataset.Cabin.isnull(), 'Cabin'] = 0


def train_models(train_df, test_df):
    # training data
    x_train = train_df.drop("Survived", axis=1)
    y_train = train_df["Survived"]
    x_test = test_df.drop("PassengerId", axis=1).copy()

    # Logistic Regression
    print("Training Logistic Regression model...")
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    model_logreg = logreg.predict(x_test)
    acc_log = logreg.score(x_train, y_train) * 100

    # Support Vector Machines
    print("Training SVM model...")
    svc = SVC()
    svc.fit(x_train, y_train)
    model_svc = svc.predict(x_test)
    acc_svc = svc.score(x_train, y_train) * 100

    # K nearest neighbors
    print("Training KNN model...")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    model_knn = knn.predict(x_test)
    acc_knn = knn.score(x_train, y_train) * 100

    # Gaussian Naive Bayes
    print("Training Gaussian Naive Bayes model...")
    gaussian = GaussianNB()
    gaussian.fit(x_train, y_train)
    model_gaussian = gaussian.predict(x_test)
    acc_gaussian = gaussian.score(x_train, y_train) * 100

    # Perceptron
    print("Training Perceptron model...")
    perceptron = Perceptron()
    perceptron.fit(x_train, y_train)
    model_perceptron = perceptron.predict(x_test)
    acc_perceptron = perceptron.score(x_train, y_train) * 100

    # Linear SVC
    print("Training Linear SVC model...")
    linear_svc = LinearSVC()
    linear_svc.fit(x_train, y_train)
    model_linear_svc = linear_svc.predict(x_test)
    acc_linear_svc = linear_svc.score(x_train, y_train) * 100

    # Stochastic Gradient Descent
    print("Training Stochastic Gradient Descent model...")
    sgd = SGDClassifier()
    sgd.fit(x_train, y_train)
    model_sgd = sgd.predict(x_test)
    acc_sgd = sgd.score(x_train, y_train) * 100

    # Decision Tree
    print("Training Decision Tree model...")
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(x_train, y_train)
    model_decision_tree = decision_tree.predict(x_test)
    acc_decision_tree = decision_tree.score(x_train, y_train) * 100

    # Random Forest
    print("Training Random Forest model...")
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(x_train, y_train)
    model_random_forest = random_forest.predict(x_test)
    acc_random_forest = random_forest.score(x_train, y_train) * 100

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
    print(
        'Selecting model ' + selected_model0['Name'] + ', ' + selected_model1['Name'] + ', ' + selected_model2['Name'])

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
    submission0.to_csv(folder + '/submission' + '_' + selected_model0['Abbrev'] + '.csv', index=False)
    submission1.to_csv(folder + '/submission' + '_' + selected_model1['Abbrev'] + '.csv', index=False)
    submission2.to_csv(folder + '/submission' + '_' + selected_model2['Abbrev'] + '.csv', index=False)


if __name__ == '__main__':
    main()

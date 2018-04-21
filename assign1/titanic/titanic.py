import time
import os

# data analysis and wrangling
import pandas as pd
import numpy as np

# machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

pd.set_option('display.width', 1000)
label = LabelEncoder()


def main():
    # reading the data
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    combine = [train_df, test_df]

    # drop ticket and cabin
    combine[0] = combine[0].drop(['Ticket', 'PassengerId'], axis=1)
    combine[1] = combine[1].drop(['Ticket'], axis=1)

    print('Original data:')
    print_preview(combine[0])

    sex_mapping = sex_to_ordinal(combine)

    fill_fare(combine[0], combine[1])
    band_fare(combine)

    fill_embarked(combine)

    transform_cabin(combine)

    create_title(combine)

    create_family_size(combine)

    fill_age(combine)

    create_child(combine)
    create_mother(combine, sex_mapping)

    band_age(combine)

    print("Final data structure:")
    print_preview(combine[0])
    print_preview(combine[1])

    combine[0].to_csv("processed_train.csv")

    train_models(combine[0], combine[1])


def print_preview(df):
    print(df.head(20))
    print('...')
    print(df.tail(20))


# sex to ordinal
def sex_to_ordinal(combine):
    for dataset in combine:
        dataset['Sex'] = label.fit_transform(dataset['Sex'])

    return dict(zip(label.classes_, label.transform(label.classes_)))


def fill_fare(train_df, test_df):
    # add missing fare, there is one missing fare with S and 3
    full = pd.concat([train_df, test_df])
    fare_guess = full.loc[(full['Embarked'] == 'S') & (full['Pclass'] == 3), 'Fare'].mean()
    test_df['Fare'] = test_df['Fare'].fillna(fare_guess)


def fill_embarked(combine):
    # fill in the missing embarked port
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna('C')


def band_fare(combine):
    # create fare band
    for dataset in combine:
        dataset['FareBand'] = pd.qcut(combine[0]['Fare'], 4)
        dataset['Fare'] = label.fit_transform(dataset['FareBand'])
        dataset.drop(['FareBand'], axis=1, inplace=True)


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

    # drop name and passangerid
    combine[0] = combine[0].drop(['Name'], axis=1)
    combine[1] = combine[1].drop(['Name'], axis=1)


def fill_age(combine):
    train_x = ['Age', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'IsAlone']
    dummy = pd.get_dummies(combine[0][train_x]).copy()

    x_age_train = dummy.loc[dummy["Age"].notnull()].drop("Age", axis=1)
    y_age_train = dummy.loc[dummy["Age"].notnull()]["Age"].astype(int)

    print("Training Decision Tree model for predicting age...")
    print(x_age_train.info())
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(x_age_train, y_age_train)

    for dataset in combine:
        test_x = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'IsAlone']
        x_age_test = pd.get_dummies(dataset[test_x]).copy()
        model_decision_tree = decision_tree.predict(x_age_test)
        dataset['PredictedAge'] = model_decision_tree
        dataset.loc[dataset['Age'].isnull(), 'Age'] = dataset['PredictedAge']
        dataset['Age'] = dataset['Age'].astype(int)

    combine[0] = combine[0].drop(['PredictedAge'], axis=1)
    combine[1] = combine[1].drop(['PredictedAge'], axis=1)


def band_age(combine):
    # creating age bands
    for dataset in combine:
        dataset['AgeBand'] = pd.cut(combine[0]['Age'], 5)

        # replace age with ordinals
        dataset['Age'] = label.fit_transform(dataset['AgeBand'])

        # remove age band feature
        dataset.drop(['AgeBand'], axis=1, inplace=True)


def create_family_size(combine):
    # create FamilySize feature from sibsp and parch
    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[(dataset['SibSp'] + dataset['Parch'] + 1) == 1, 'IsAlone'] = 1


def create_child(combine):
    for dataset in combine:
        dataset['Child'] = 0
        dataset.loc[dataset['Age'] < 18, 'Child'] = 1


def create_mother(combine, sex_mapping):
    for dataset in combine:
        dataset['Mother'] = 0
        dataset.loc[(dataset['Child'] == 0) & (dataset['Parch'] > 0) & (dataset['Sex'] == sex_mapping['female']) &
                    (dataset['Title'] != 'Miss'), 'Mother'] = 1


def transform_cabin(combine):
    for dataset in combine:
        dataset.loc[dataset.Cabin.notnull(), 'Cabin'] = dataset['Cabin'].astype(str).str[0]
        dataset.loc[dataset.Cabin.isnull(), 'Cabin'] = 'X'


def train_model(alg, name, abbrev, x_train, y_train, x_test, ids, folder):
    alg.fit(x_train, y_train)
    model = alg.predict(x_test)
    acc = (cross_val_score(alg, x_train, y_train))

    submission = pd.DataFrame({
        "PassengerId": ids,
        "Survived": model
    })

    submission.to_csv(folder + '/submission' + '_' + abbrev + '.csv', index=False)

    print('\\textbf{index} & ' + name + ' & ' + str(np.mean(acc)) + ' & ' + str(max(acc) - min(acc)) + ' \\\\')


def train_models(train_df, test_df):
    # training data
    x = ['Age', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'IsAlone']
    x_train = pd.get_dummies(train_df[x]).copy()
    y_train = train_df["Survived"]
    test_df = pd.get_dummies(test_df[x + ["PassengerId"]]).copy()
    # test_df['Cabin_T'] = 0

    x_test = test_df.drop(["PassengerId"], axis=1)
    x_test = x_test[x_train.columns.values]

    folder = 'results/' + str(round(time.time()))
    os.makedirs(folder)

    # Logistic Regression
    train_model(LogisticRegression(), 'Logistic Regression', 'LOG_REG', x_train, y_train, x_test,
                test_df["PassengerId"], folder)

    # Support Vector Machines
    train_model(SVC(), 'Support Vector Machine', 'SVM', x_train, y_train, x_test,
                test_df["PassengerId"], folder)

    # K nearest neighbors
    train_model(KNeighborsClassifier(n_neighbors=3), 'KNN', 'KNN', x_train, y_train, x_test,
                test_df["PassengerId"], folder)

    # Gaussian Naive Bayes
    train_model(GaussianNB(), 'Gaussian Naive Bayes', 'GNB', x_train, y_train, x_test,
                test_df["PassengerId"], folder)

    # Perceptron
    train_model(Perceptron(), 'Perceptron', 'PERC', x_train, y_train, x_test,
                test_df["PassengerId"], folder)

    # Linear SVC
    train_model(LinearSVC(), 'Linear SVC', 'LIN_SVC', x_train, y_train, x_test,
                test_df["PassengerId"], folder)

    # Stochastic Gradient Descent
    train_model(SGDClassifier(), 'Stochastic Gradient Descent', 'SGD', x_train, y_train, x_test,
                test_df["PassengerId"], folder)

    # Decision Tree
    train_model(DecisionTreeClassifier(), 'Decision Tree', 'D_TREE', x_train, y_train, x_test,
                test_df["PassengerId"], folder)

    # Random Forest
    train_model(RandomForestClassifier(n_estimators=100), 'Random Forest', 'RF', x_train, y_train, x_test,
                test_df["PassengerId"], folder)

    # l = zip(x_train.columns.values, list(random_forest.feature_importances_))
    # asd = [i for i in l]
    # asd.sort(key=lambda x: x[1], reverse=True)
    # for i in asd:
    #     print(i)

    train_model(ExtraTreesClassifier(n_estimators=300, max_features=None), 'Extra Trees', 'ET', x_train, y_train,
                x_test, test_df["PassengerId"], folder)

    train_model(GradientBoostingClassifier(n_estimators=100), 'Gradient Boosting', 'GB', x_train, y_train, x_test,
                test_df["PassengerId"], folder)


if __name__ == '__main__':
    main()

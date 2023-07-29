import pandas as pnds
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree

train = pnds.read_csv('Train.csv')
test = pnds.read_csv('Test.csv')
train_test_data = [train, test]  # combining train and test dataset

# ************* clean names **************
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # summarize the name to get the title only

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2,
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3, "Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona": 3, "Mme": 3, "Capt": 3, "Sir": 3}
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    # Title map: Mr : 0 , Miss : 1 , Mrs: 2 , Others: 3

# train.drop('Name', axis=1, inplace=True)  # delete unnecessary feature from dataset
# test.drop('Name', axis=1, inplace=True)

# ************* transform sex **************
sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
    # Sex map: Male : 0 , Female : 1

# ************* clean ages **************
# fill missing age with median age for each title
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)

# ************* clean embarks **************
# fill out missing embark with mode embark of all values
embark = list(train.iloc[:, 10].mode())[0]
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna(embark)

embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
    # Embark map: S : 0 , C : 1 , Q : 2

# ************* clean cabins **************
# fill missing cabin with median cabin for each Pclass
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]

cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

# print(train.head())
# print(train.info())
# ---------------------------------------------------------------------------------------------------------------


train.drop(['PassengerId',
            'Name',
            'SibSp',
            'Parch',
            'Ticket',
            'Cabin',
            'Embarked',
            'Title'],
           axis='columns',inplace=True)

inputs = train.drop('Survived', axis='columns')
target = train.Survived



x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2)

model = tree.DecisionTreeClassifier(max_depth=3)
dt_model = model.fit(x_train, y_train)

fig = plt.figure(figsize=(12, 10))
tree.plot_tree(dt_model, feature_names=['Pclass','Sex','Age','Fare'],
               class_names=['Not survived', 'Survived'])
#plt.show()
print(train.head())
print(dt_model.score(x_test,y_test))
#The output is 0.8100558659217877
#That means 81.01% of the training set is correctly classified
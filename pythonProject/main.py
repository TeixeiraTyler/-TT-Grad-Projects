import pandas as pd
import random
import math
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 1000)

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]

# 1 print(train_df.columns)

# 2 ‘Name’, ‘Sex’, ‘Ticket’, ‘Cabin’, ‘Embarked’

# 3 'PassengerId', ‘Survived’, 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'

# 4 'Ticket’

# 5 Training: [‘Age’, ‘Cabin’, ‘Embarked’] 		Test: [‘Age’, ‘Cabin’]

# 6 int, int, int, String, String, float, int, int, String, float, String, String

# 7 print(train_df.describe())

# 8 print(train_df.describe(include=object))

# 9 print(train_df.groupby(['Pclass']).mean().sort_values('Survived', ascending=False))

# 10 print(train_df.groupby(['Sex']).mean().sort_values('Survived', ascending=False))

# 11
# age_lived_df = train_df.loc[train_df['Survived'] == 1]
# age_died_df = train_df.loc[train_df['Survived'] == 0]
# age_lived_df.hist(column='Age')
# age_died_df.hist(column='Age')
# plt.show()

# 12
# class1_lived_df = train_df.loc[(train_df['Survived'] == 1) & (train_df['Pclass'] == 1)]
# class2_lived_df = train_df.loc[(train_df['Survived'] == 1) & (train_df['Pclass'] == 2)]
# class3_lived_df = train_df.loc[(train_df['Survived'] == 1) & (train_df['Pclass'] == 3)]
# class1_died_df = train_df.loc[(train_df['Survived'] == 0) & (train_df['Pclass'] == 1)]
# class2_died_df = train_df.loc[(train_df['Survived'] == 0) & (train_df['Pclass'] == 2)]
# class3_died_df = train_df.loc[(train_df['Survived'] == 0) & (train_df['Pclass'] == 3)]
# class1_lived_df.hist(column='Age')
# class2_lived_df.hist(column='Age')
# class3_lived_df.hist(column='Age')
# class1_died_df.hist(column='Age')
# class2_died_df.hist(column='Age')
# class3_died_df.hist(column='Age')
# plt.show()

#13
# embarkS_lived_df = train_df.loc[(train_df['Survived'] == 1) & (train_df['Embarked'] == 'S')]
# embarkC_lived_df = train_df.loc[(train_df['Survived'] == 1) & (train_df['Embarked'] == 'C')]
# embarkQ_lived_df = train_df.loc[(train_df['Survived'] == 1) & (train_df['Embarked'] == 'Q')]
# embarkS_died_df = train_df.loc[(train_df['Survived'] == 0) & (train_df['Embarked'] == 'S')]
# embarkC_died_df = train_df.loc[(train_df['Survived'] == 0) & (train_df['Embarked'] == 'C')]
# embarkQ_died_df = train_df.loc[(train_df['Survived'] == 0) & (train_df['Embarked'] == 'Q')]
# embarkS_lived_df.hist(column='Fare', by='Sex')
# embarkC_lived_df.hist(column='Fare', by='Sex')
# embarkQ_lived_df.hist(column='Fare', by='Sex')
# embarkS_died_df.hist(column='Fare', by='Sex')
# embarkC_died_df.hist(column='Fare', by='Sex')
# embarkQ_died_df.hist(column='Fare', by='Sex')
# plt.show()

#14 print(train_df.groupby(['Ticket']).mean().sort_values('Survived', ascending=False))

#15 print(train_df['Cabin'].isnull().sort_values())

#16
# train_df['Gender'] = train_df['Sex'].rank(method='dense', ascending=False).astype(int)
# re = {1: 0, 2: 1}
# train_df['Gender'].replace(re, inplace=True)
# train_df.drop(columns=['Sex'])
# print(train_df)

#17
# avg_low = int(train_df['Age'].mean()) - 15
# avg_high = int(train_df['Age'].mean() + 15)
# for index, row in train_df.iterrows():
#     if math.isnan(row['Age']):
#         train_df.loc[index, 'Age'] = random.randint(avg_low, avg_high)
# print(train_df)

#18
# train_df['Embarked'].fillna('S')
# print(train_df)

#19
# val = train_df['Fare'].mode()
# train_df = train_df.replace(['Fare'], val)
# print(train_df)

#20
# for index, row in train_df.iterrows():
#     if row['Fare'] <= 7.91:
#         train_df.loc[index, 'Fare'] = 0
#     if (row['Fare'] > 7.91) & (row['Fare'] < 14.454):
#         train_df.loc[index, 'Fare'] = 1
#     if (row['Fare'] >= 14.454) & (row['Fare'] < 31.0):
#         train_df.loc[index, 'Fare'] = 2
#     if row['Fare'] > 31.0:
#         train_df.loc[index, 'Fare'] = 3
# print(train_df)

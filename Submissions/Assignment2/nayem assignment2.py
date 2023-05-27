import pandas as pd
from sklearn.metrics import (accuracy_score as acc,
    precision_score as prec,recall_score as recall,
    f1_score as f1,confusion_matrix as cf,
    precision_recall_fscore_support as support)
from sklearn.model_selection import (KFold as KF,
    train_test_split as split)
from sklearn.linear_model import LogisticRegression as LgR

df = pd.read_csv('lung_cancer.csv')
# print(df)

df['gender'] = df['gender'] == 'Male'
# print(df)

X = df['race']

race = []
for x in X:
    if (x not in race and not x != x):
        race.append(x)

# print(race)

def assign_matching_indices(target_list, distinct_elements):
    index_mapping = {
        element: index for index, element in enumerate(distinct_elements)
    }
    assigned_indices = []
    for element in target_list:
        if element != element:
            assigned_indices.append(-1)
        else:
            assigned_indices.append(index_mapping[element])

    return assigned_indices


df['race'] = assign_matching_indices(df['race'],race)

# print(df)

df['smoker'] = df['smoker'] == 'Current'

# print(df)

df = df.dropna()

# print(df.shape)

# df['malignent'] = df['days_to_cancer'] > 0

# print(df)

# print((df['malignent'] == True).count)

# X = df[['age','gender','race','smoker']].values
# y = df['malignent'].values

# X_train, X_test, y_train, y_test = split(X, y,train_size=0.7)

# model = LgR()
# model.fit(X_train,y_train)

# y_pred = model.predict(X_test)

# print("Accuaracy: ",acc(y_test,y_pred))

# print("Precision: ",prec(y_test,y_pred))

# print("Recall: ",recall(y_test,y_pred))

# print("F1 score: ",f1(y_test,y_pred))




# Let's creat negative classes!
# Let's creat negative classes!

# Let's creat negative classes!
# Let's creat negative classes!

X = df['stage_of_cancer']

stage = []
for x in X:
    if (x not in stage):
        stage.append(x)

# print(stage)

df['stage_of_cancer'] = assign_matching_indices(df['stage_of_cancer'],stage)

# print(df['stage_of_cancer'])

array = []
for x in df['stage_of_cancer']:
    if(x == 0 or x == 6):
        array.append(False)
    else:
        array.append(True)

# print(array)

df['malignent2'] = array

# print(df['malignent2'])


X2 = df[['age','gender','race','smoker']].values
y2 = df['malignent2'].values

X_train, X_test, y_train, y_test = split(X2, y2,train_size=0.7)

model = LgR()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Accuaracy: ",acc(y_test,y_pred))

print("Precision: ",prec(y_test,y_pred))

print("Recall: ",recall(y_test,y_pred))

print("F1 score: ",f1(y_test,y_pred))

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,precision_recall_fscore_support


df = pd.read_csv('../titanic.csv')

df['Male'] = df['Sex'] == 'male'


X = df[['Pclass','Male','Age','Siblings/Spouses','Parents/Children','Fare']].values

y = df['Survived'].values


model = LogisticRegression()

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.8,random_state = 5)

print("Whole dataset :",X.shape)
print("Training set :",X_train.shape)
print("Test set :",X_test.shape)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

passenger = [[1,1,21,0,4,10]]

#prediction

print(model.predict(passenger))

print(model.predict(X[:5]))

print(y[:5])

print(y_pred)

mask = y_pred == y

print(y[mask].sum()/y.shape[0])

print(confusion_matrix(y_test,y_pred))

print(precision_score(y_test,y_pred))

print(recall_score(y_test,y_pred))

print(f1_score(y_test,y_pred))

print("accuracy :",accuracy_score(y_test,y_pred))
print("precision :",precision_score(y_test,y_pred))
print("recall :",recall_score(y_test,y_pred))
print("f1 score :",f1_score(y_test,y_pred))


sensitivity_score = recall_score


def specificity_score(y_true,y_pred):
    p,r,f,s = precision_recall_fscore_support(y_true,y_pred)
    return r[0]

def sensitivity_score2(y_true,y_pred):
    p,r,f,s = precision_recall_fscore_support(y_true,y_pred)
    return r[1]

#r[0], recall of the negative class


print(sensitivity_score(y_test,y_pred))

print(sensitivity_score2(y_test,y_pred))

print(specificity_score(y_test,y_pred))


threshold = 0.80
y_pred3 = model.predict_proba(X_test)[:,1] > threshold
print(y_pred3)

print("precision",precision_score(y_test,y_pred3))


y_pred_proba = model.predict_proba(X_test)

fpr,tpr,thresholds = roc_curve(y_test,y_pred_proba[:,1])


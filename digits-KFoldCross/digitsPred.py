#K-Fold Cross Validation
#%%
from sklearn.linear_model import LogisticRegression
import pandas as pd 
from matplotlib import pyplot as plt 
from sklearn.svm import SVC 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
# %%
digits = load_digits()
# %%
from sklearn.model_selection import train_test_split as tts 
X_train, X_test, y_train, y_test = tts(digits.data, digits.target, test_size = 0.3)
# %%
lr = LogisticRegression()
lr.fit(X_train,y_train)
lr.score(X_test, y_test)
# %%
svm = SVC()
svm.fit(X_train,y_train)
svm.score(X_test,y_test)
# %%
rf = RandomForestClassifier(n_estimators=40)
rf.fit(X_train, y_train)
rf.score(X_test,y_test)
# %%
#k-fold
from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
# %%
kf
# %%
#example
for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index, test_index)
# %%
def get_score(model,X_train,X_test,y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test,y_test)
# %%
get_score(LogisticRegression(),X_train,X_test,y_train, y_test)
# %%
#usually used 10
from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits=3)

# %%
scores_logistic = []
scores_svm = []
scores_rf = []

for train_index, test_index in folds.split(digits.data,digits.target):
    X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], \
                                       digits.target[train_index], digits.target[test_index]
    scores_logistic.append(get_score(LogisticRegression(solver='liblinear',multi_class='ovr'), X_train, X_test, y_train, y_test))  
    scores_svm.append(get_score(SVC(gamma='auto'), X_train, X_test, y_train, y_test))
    scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test))

# %%
scores_logistic
# %%
scores_rf
# %%
scores_svm
# %%
from sklearn.model_selection import cross_val_score
# %%
#same line of code in 48 but comprised
#using cross validation
cross_val_score(LogisticRegression(),digits.data, digits.target)
# %%
cross_val_score(SVC(),digits.data, digits.target)
#%%
cross_val_score(RandomForestClassifier(n_estimators=40),digits.data, digits.target)
# %%

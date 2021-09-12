#%%
#Naive Bayes- Features are not inter related (assumption)
import pandas as pd 
df = pd.read_csv('titanic.csv')
# %%
df
#%%
df.columns
# %%
df=df.drop(['PassengerId','Name', 'SibSp', 'Parch',
       'Ticket', 'Cabin', 'Embarked'],axis=1)
# %%
df
# %%
target = df.Survived
# %%
inputs = df.drop('Survived',axis=1)
# %%
dummies = pd.get_dummies(inputs.Sex)
# %%
dummies.head()
# %%
inputs = pd.concat([inputs,dummies],axis=1)
# %%
inputs
# %%
inputs = inputs.drop('Sex',axis=1)
# %%
inputs
# %%
inputs.columns[inputs.isna().any()]
# %%
inputs.Age[:10]
# %%
inputs.Age = inputs.Age.fillna(inputs.Age.mean())
# %%
inputs
# %%
from sklearn.model_selection import train_test_split

# %%
X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)
# %%
from sklearn.naive_bayes import GaussianNB
#Gaussian is used when there is normal distribution
# %%
model = GaussianNB()
# %%
model.fit(X_train, y_train)
# %%
model.score(X_test, y_test)
# %%
X_test[:10]
# %%
y_test[:10]
# %%
model.predict(X_test[:10])
# %%
model.predict_proba(X_test[:10])
# %%

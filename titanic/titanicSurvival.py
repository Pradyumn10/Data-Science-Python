#%%
import pandas as pd 
# %%
titanic = pd.read_csv("train.csv")
# %%
titanic
# %%
#classification problem as we have finite sets to predict
#we use stratify in this model
y=titanic["Survived"]
# %%
X=titanic.drop('Survived',axis=1)
# %%
X
# %%
#removing unwanted columns
X = X.drop(['PassengerId','Name','Ticket','Embarked','Cabin'], axis=1)
# %%
X
# %%
X.isna().sum()
# %%
y.isna().sum()
# %%
X['Age']=X['Age'].fillna(X['Age'].mean())
# %%
X.isna().sum()
# %%
X.dtypes
# %%
#using get dummies method as we have 2 values in single column. It will delete one column as we used drop_first = T ;
X=pd.get_dummies(X,columns=['Sex'], drop_first=True)
# %%
X
# %%
type(X)
# %%
from sklearn.model_selection import train_test_split as tts
# %%
X_train, X_test, y_train, y_test = tts(X,y, random_state=42, stratify=y)
# %%
X
# %%
#Features are not in same scale
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# %%
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# %%
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# %%
model.fit(X_train, y_train)
# %%
model.score(X_test, y_test)
# %%
predData = pd.DataFrame({0:[1],1:[22],2:[2],3:[3],4:[71],5:[0]})
# %%
predData
# %%
predData = scaler.transform(predData)
# %%
model.predict(predData)
# %%

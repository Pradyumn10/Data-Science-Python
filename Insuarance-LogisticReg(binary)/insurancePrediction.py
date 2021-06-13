#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# %%
df = pd.read_csv('./insurance_data.csv')
# %%
df
# %%
plt.scatter(df.age, df.bought_insurance, marker='+', color='red')
# %%
X_train,X_test, y_train,y_test = train_test_split(df[['age']],df.bought_insurance,test_size=0.1)
# %%
X_test
# %%
from sklearn.linear_model import LogisticRegression
# %%
model = LogisticRegression()
# %%
model.fit(X_train, y_train)
# %%
model.coef_
# %%
model.intercept_
# %%
pred = model.predict(X_test)
# %%
model.score(X_test,y_test)
# %%
model.predict_proba(X_test)
# %%

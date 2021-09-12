#%%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model
# %%
#REading csv file
df = pd.read_csv("homeprices.csv")
# %%
df
# %%
#Visaulization-scatter plot
plt.xlabel("Area(sq. ft.")
plt.ylabel("Price(US $)")
plt.scatter(df.area,df.price , color='red',marker='+')
# %%
#linear model follows linear equation : y=mx+c
reg = linear_model.LinearRegression()
# %%
reg.fit(df[['area']],df.price)
# %%
reg.predict([[3300]])
# %%
#Finding values of linear equation
#value of m
reg.coef_
# %%
#value of intercept
reg.intercept_
#linear model equation : price=135.78*area+180616
#135.787*3300+180616=628715.75342466
# %%
#plotting the best fit line
plt.xlabel("Area(sq. ft.")
plt.ylabel("Price(US $)")
plt.scatter(df.area,df.price , color='red',marker='+')
plt.plot(df.area, reg.predict(df[['area']]),color='blue')

#%%
#test file
df2 = pd.read_csv("areas.csv")
# %%
df2
# %%
pred = reg.predict(df2)
# %%
df2['prices']=pred
# %%
#df2.to_csv('predictions.csv',index=False)
# %%
#Saving model using pickle and joblib
import pickle
# %%
with open('model_pickle','wb') as f:
    pickle.dump(reg,f)
# %%
with open('model_pickle','rb') as f:
    mp = pickle.load(f)
# %%
mp.predict([[3300]])
# %%
import joblib
# %%
joblib.dump(reg,'model_joblib')
# %%
mj = joblib.load('model_joblib')
# %%
mj.predict([[3300]])
# %%
mj.coef_
# %%

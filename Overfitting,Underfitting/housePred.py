#%%
import pandas as pd 
from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split
# %%
dataset = pd.read_csv('Melbourne_housing_FULL.csv')
# %%
dataset
# %%
#unqiue value
dataset.nunique()
# %%
dataset.shape
#%%
dataset.columns
# %%
cols_to_use = ['Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Regionname', 
               'Propertycount', 'Distance', 'CouncilArea', 'Bedroom2',
               'Bathroom', 'Car', 'Landsize', 'BuildingArea','Price']
dataset = dataset[cols_to_use]
# %%
dataset
# %%
dataset.shape
# %%
dataset.isna().sum()
# %%
cols_to_fill_zero = ['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']
dataset[cols_to_fill_zero] = dataset[cols_to_fill_zero].fillna(0)
# %%
dataset.isna().sum()
# %%
dataset['Landsize'] = dataset['Landsize'].fillna(dataset['Landsize'].mean())
dataset['BuildingArea'] = dataset['BuildingArea'].fillna(dataset['BuildingArea'].mean())
# %%
dataset.dropna(inplace=True)
# %%
dataset.isna().sum()
# %%
dataset = pd.get_dummies(dataset, drop_first=True)
# %%
dataset
# %%
X=dataset.drop('Price',axis=1)
y=dataset['Price']
# %%
train_X, test_x, train_y , test_y = train_test_split(X,y, test_size=0.3, random_state=2)
# %%
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(train_X,train_y)
# %%
reg.score(test_x,test_y)
# %%
reg.score(train_X,train_y)
#Overfitting
# %%
#overvcoming it with Lasso Regression (L1)
from sklearn import linear_model
lasso_reg = linear_model.Lasso(alpha=50, max_iter=100,tol=0.1)
lasso_reg.fit(train_X,train_y)
# %%
lasso_reg.score(test_x,test_y)
# %%
lasso_reg.score(train_X,train_y)
# %%
#L2
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=50,max_iter=100,tol=0.1)
ridge_reg.fit(train_X,train_y)
# %%
ridge_reg.score(test_x,test_y)
# %%
ridge_reg.score(train_X,train_y)
# %%

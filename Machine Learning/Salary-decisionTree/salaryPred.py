#%%
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
# %%
df = pd.read_csv('./salaries.csv')
# %%
df
# %%
inputs = df.drop('salary_more_then_100k',axis=1)
# %%
target = df['salary_more_then_100k']
# %%
target
# %%
from sklearn.preprocessing import  LabelEncoder
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()
# %%
inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_company.fit_transform(inputs['job'])
inputs['degree_n'] = le_company.fit_transform(inputs['degree'])

# %%
inputs.head()
# %%
inputs_n = inputs.drop(['company','job','degree'],axis=1)
# %%
inputs_n
# %%
from sklearn import tree
model = tree.DecisionTreeClassifier()
# %%
model.fit(inputs_n, target)
# %%
model.score(inputs_n, target)
# %%

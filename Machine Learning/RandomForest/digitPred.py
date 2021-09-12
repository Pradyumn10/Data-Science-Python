#%%
import pandas as pd 
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
digits = load_digits()
# %%
dir(digits)
# %%
plt.gray()
for i in range(4):
    plt.matshow(digits.images[i])
# %%
df = pd.DataFrame(digits.data)
# %%
df
# %%
df['target'] = digits.target
# %%
from sklearn.model_selection import train_test_split
# %%
X_train,X_test, y_train,y_test = train_test_split(df.drop(['target'],axis=1), digits.target, test_size=0.2)
# %%
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=30)
# %%
model.fit(X_train, y_train)
# %%
model.score(X_test, y_test)
#%%
y_predicted = model.predict(X_test)
#%%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predicted)
# %%
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# %%

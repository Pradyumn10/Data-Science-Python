#%%
import pandas as pd 
from matplotlib import pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
# %%
df = pd.read_csv('income.csv')
# %%
df
# %%
plt.scatter(df['Age'],df['Income($)'])
# %%
km = KMeans(n_clusters=3)
# %%
km
# %%
y_pred = km.fit_predict(df[['Age','Income($)']])
# %%
y_pred
# %%
df['cluster'] = y_pred
df
# %%
df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.scatter(df1.Age, df1['Income($)'],color='green')
plt.scatter(df2.Age, df2['Income($)'],color='red')
plt.scatter(df3.Age, df3['Income($)'],color='black')

plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()
# %%
scaler = MinMaxScaler()
# %%
scaler.fit(df[['Income($)']])
df[['Income($)']] = scaler.transform(df[['Income($)']])

# %%
df
# %%
scaler.fit(df[['Age']])
df[['Age']] = scaler.transform(df[['Age']])
# %%
df
# %%
y_predicted = km.fit_predict(df[['Age','Income($)']])
# %%
df['cluster1'] = y_predicted
df.drop('cluster',axis=1,inplace=True)
# %%
df
#%%
km.cluster_centers_
# %%
df1 = df[df.cluster1==0]
df2 = df[df.cluster1==1]
df3 = df[df.cluster1==2]

plt.scatter(df1.Age, df1['Income($)'],color='green',label='Cluster 1')
plt.scatter(df2.Age, df2['Income($)'],color='red', label='Cluster 2')
plt.scatter(df3.Age, df3['Income($)'],color='black', label='Cluster 3')
plt.scatter(km.cluster_centers_[:,0],
km.cluster_centers_[:,1], color='purple', marker='+', label='centroid')

plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()

# %%
#elbow plot
k_rng = range(1,10)
sse=[]
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age','Income($)']])
    sse.append(km.inertia_)
# %%
sse
# %%
#Elbow plot
plt.plot(k_rng,sse)
plt.xlabel('K')
plt.ylabel('Sum of Squared Error')
# %%

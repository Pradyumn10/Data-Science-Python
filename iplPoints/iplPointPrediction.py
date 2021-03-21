#%%
'''
* @FileName : iplPointPrediction.py
* @Author : Pradyumn Joshi
* @Brief : Given details of ipl teams and players predict the total points they scored on the basis of runs, balls,wickets
* @Date : 24 Jan 2021
*
* Copyright (C) 2021
'''
#%%
'''
1. Extract Features
    a. Features and target should not have null value
    b. Features should be numeric in nature
    c. Features should be of the type array/dataframe
    d. Features should have some rows and columns  
2. Split the dataset into training and testing datasets.
    e. Features should be on same scale
3. Train the model on training dataset
4. Test the model on testing dataset
'''
#%%
import pandas as pd
ipl = pd.read_csv("ipl2017.csv")
#%%
ipl
# %%
y=ipl["total"]
# %%
X=ipl.drop(["total"],axis=1)
# %%
#checking null values
X.isna().sum()
# %%
y.isna().sum()
# %%
X.shape
# %%
X.dtypes
#%%
X
#%%
X=X.drop(['mid','date'],axis=1)
# %%
#encoding the data (object-> numeric)
from sklearn.preprocessing import LabelEncoder
l_encoder = LabelEncoder()
#%%
l_venue = X['venue']
l_bat = X['bat_team']
l_bowl = X['bowl_team']
l_batsman = X['batsman']
l_bowler = X['bowler']
#%%
l_encoder.fit(X['venue'])
X['venue']=l_encoder.transform(X['venue'])
#%%
#decoded value with encoded value
lVen = dict(zip(l_venue,X['venue']))
# %%
l_encoder.fit( X['bat_team'] )
X['bat_team']=l_encoder.transform(X['bat_team'])
#%%
lBat = dict(zip(l_bat,X['bat_team']))
# %%
l_encoder.fit( X['bowl_team'] )
X['bowl_team']=l_encoder.transform(X['bowl_team'])
#%%
lBowl = dict(zip(l_bowl,X['bowl_team']))
# %%
l_encoder.fit( X['batsman'] )
X['batsman']=l_encoder.transform(X['batsman'])
#%%
lBatsman = dict(zip(l_batsman, X['batsman']))
# %%
l_encoder.fit( X['bowler'] )
X['bowler']=l_encoder.transform(X['bowler'])
#%%
lBowler = dict(zip(l_bowler,X['bowler']))
# %%
X
# %%
from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X,y, test_size=0.25, random_state=42)
# %%
#scalling the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# %%
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
# %%
model.fit(X_train,y_train)
#%%
model.score(X_test,y_test)
# %%
pred_data = pd.DataFrame({'venue':[14],'bat_team':[7], 'bowl_team':[12], 'batsmen':[224], 'bowler':[192] ,'runs' : [100], 'wickets': [0], 'overs':[2.0], 'runs_last_5': [44], 'wickets_last_5': [0], 'striker': [0], 'non-striker':[2]})
# %%
pred_data
# %%
model.predict(pred_data)
#%%
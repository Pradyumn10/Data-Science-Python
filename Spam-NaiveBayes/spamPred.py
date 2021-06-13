#%%
import pandas as pd  
# %%
df = pd.read_csv('spam.csv')
# %%
df
# %%
df.groupby('Category').describe()
# %%
df['spam'] = df['Category'].apply(lambda x:1 if x=='Spam' else 0)
# %%
df
# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df.spam, test_size=0.25)
# %%
#using count vectorizer technique - counting occurance of a word
from sklearn.feature_extraction.text import CountVectorizer
# %%
v=CountVectorizer()
# %%
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()[:3]
# %%
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
# %%
model.fit(X_train_count, y_train)
# %%
emails = [
    'Hey Mohan, can we get together to watch football game tomorrow?',
    'Upto 20% discount on parking, exlusive offer just for you. Dont miss the reward! '
]
emails_count = v.transform(emails)
# %%
model.predict(emails_count)
# %%
X_test_count = v.transform(X_test)
# %%
model.score(X_test_count,y_test)
# %%
#creating pipeline
from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('Vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
# %%
clf.fit(X_train, y_train)
# %%
clf.score(X_test,y_test)
# %%

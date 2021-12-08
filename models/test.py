import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import nltk
nltk.download(['punkt','stopwords'])
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from nltk.stem.porter import *


#%%
engine = create_engine('sqlite:///data/db.db')
df = pd.read_sql_table('InsertTableName',engine)
X = df['message']
Y = df.iloc[:, 4:]
del Y['child_alone']


#%%
def tokenize(text):
    lemmatizer = WordNetLemmatizer()
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # Split text into words using NLTK
    words = word_tokenize(text)
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    # Reduce words to their root form
    stemmed = [PorterStemmer().stem(w) for w in words]

    clean = [lemmatizer.lemmatize(t) for t in stemmed]
    return clean



#%%
pipeline = Pipeline([
            ('vect', TfidfVectorizer(tokenizer=tokenize)),
            ('clf', MultiOutputClassifier(estimator=LogisticRegression()))
                ])
parameters = {'C': [0.1,1,10]}

#%%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
#%%
cv = GridSearchCV(pipeline, parameters,cv=5)

#%%
cv.fit(X_train, Y_train)
#%%

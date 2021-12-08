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


def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('InsertTableName',engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    del Y['child_alone']
    return X , Y , Y.columns.values


def tokenize(text):
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

    return stemmed



def build_model():
    pipeline = Pipeline([
                ('vect', TfidfVectorizer(tokenizer=tokenize)),
                ('clf', MultiOutputClassifier(estimator=LogisticRegression()))
                    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = pd.DataFrame(model.predict(X_test),columns=Y_test.columns.get_values())
    print(classification_report(np.hstack(Y_test), np.hstack(y_pred)))
    pass


def save_model(model, model_filepath):
    joblib.dump(model,f'{model_filepath}')
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

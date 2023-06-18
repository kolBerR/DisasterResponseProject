# import libraries

import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import pickle
import pandas as pd
from sqlalchemy import create_engine, text

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


def load_data(database_filepath):
    #create sql engine
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    # connect to database and load data
    q = 'SELECT * FROM drTable'
    with engine.connect() as conn:
        query = conn.execute(text(q))
    df = pd.DataFrame(query.fetchall())

    # create labels and target variables
    X = df['message']
    Y = df.drop(['original','genre','message','offer','request'], axis=1)
  
    category_names = Y.columns.values
    
    return X, Y, category_names            


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(n_estimators=8, max_depth=6))
    ])

    parameters = {
    'vect__ngram_range': ((1, 1), (1, 2)),
    'clf__n_estimators': [6, 10]
    }
    cv_model = GridSearchCV(pipeline, param_grid=parameters)

    return cv_model


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred=model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))
    print(pipeline.get_params())


def save_model(model, model_filepath):
    pickle.dump(model, open('{}'.format(model_filepath), 'wb'))


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
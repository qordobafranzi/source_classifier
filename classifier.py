import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import numpy as np
from sklearn import naive_bayes as nb



def main():
    langs = []
    raw_text = []
    y_train = []
    scores = []
    accuracy = []
    confusion = np.array([[0, 0], [0, 0]])
    # Reading samples as training data
    for subdir, _, files in os.walk('./samples'):
        if subdir == "./samples":
            print "skip" + subdir
            continue

        lang = subdir.split("/")[-1]
        langs.append(lang)

        label = langs.index(lang)

        for f in files:
            y_train.append(label)
            with open(subdir + "/" + f, "r") as lang_file:
                raw_text.append(lang_file.read())

    
    X = np.array(raw_text)
    y = np.array(y_train)
    kf = KFold(n_splits=2)
    kf.get_n_splits(X)

    pipeline = Pipeline([
        # tokenization and occurrence counting in a single class ( 2-grams of words in addition to the 1-grams)
        ('vectorizer',  CountVectorizer(ngram_range=(1, 2))),
        # the term frequency, the number of times a term occurs in a given document, is multiplied with idf component
        # ('tfidf_transformer',  TfidfTransformer()),
        ('classifier',  nb.MultinomialNB()) ])

    KFold(n_splits=3, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        accuracy_status = accuracy_score(y_test, predictions)
        accuracy.append(accuracy_status)

        with open('../railsapp/app/assests') as lang_file:
                raw_text.append(lang_file.read())
        predict_example = pipeline.predict(example)
    
    print('Total classified:', len(y_train))
    print('accuracy', sum(accuracy)/len(accuracy))
    # print(predictions)

 
if __name__ == '__main__':
    main()
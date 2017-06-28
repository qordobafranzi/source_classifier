import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import naive_bayes as nb
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes as nb
import pandas as pd
from binaryornot.check import is_binary
import pprint

def main():
    pipeline = Pipeline([
        ('vectorizer',  CountVectorizer(ngram_range=(1, 2))),
        # ('tfidf_transformer',  TfidfTransformer()),
        ('classifier',  nb.MultinomialNB())])

    langs = []
    X_train = []
    y_train = []

    # Reading samples as training data
    for subdir, _, files in os.walk('./samples'):
        if subdir == "./samples":
            print "skip" + subdir
            continue

        lang = subdir.split("/")[-1]
        langs.append(lang)
        label = langs.index(lang)

        for f in files:
            if '.DS_Store' in str(files):
                pass
            else:
                y_train.append(lang)
                with open(subdir + "/" + f, "r") as lang_file:

                    X_train.append(lang_file.read())

    print(len(y_train))
    print(len(X_train))
    pipeline.fit(X_train, y_train)

    Examples = []
    Files = []
    for pred_subdir, dir_, pred_files in os.walk('../rails_admin/app'):
        if pred_subdir == "../rails_admin/app":
            print "skip" + pred_subdir
            continue
        for f_ in pred_files:
            if is_binary(pred_subdir + "/" + f_):
                pass
            elif not is_binary(pred_subdir + "/" + f_):
                with open(pred_subdir + "/" + f_, "r") as pred_file:
                    Examples.append(pred_file.read())
                    Files.append(f_)
            else:
                print("hi")

    predict_examples = pipeline.predict(Examples)
    classifier = dict()
    for i in predict_examples:
        classifierExamples[i]


    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(predict_examples)
    pp.pprint(Files)

if __name__ == '__main__':
    main()
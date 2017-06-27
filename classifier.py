import os
from sklearn.feature_extraction.text import CountVectorizer


def main():
    langs = []
    raw_text = []
    y_train = []
    count_vect = CountVectorizer(ngram_range=(1, 2))

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

    X_train = count_vect.fit_transform(raw_text)


    # Split test data from samples
    from sklearn import cross_validation as cv
    X_train, X_test, y_train, y_test = cv.train_test_split(X_train, y_train)


    # raw_text_test = []
    # y_test = []
    # for subdir_test, _, files_test in os.walk('samples'):
    #     print("subdir_test {}".format(subdir_test))
    #     if subdir_test == "samples":
    #         print "skip" + subdir_test
    #         continue

    #     lang_test = subdir_test.split("/")[-1]
    #     # print("lang_test {}".format(lang_test))
    #     label_test = langs.index(lang_test)
    #     # print("label_test {}".format(label_test))

    #     for f in files_test:
    #         y_test.append(label_test)
    #         with open(subdir_test + "/" + f, "r") as lang_file:
    #             raw_text_test.append(lang_file.read())
    # X_test = count_vect.transform(raw_text_test)

    from sklearn import naive_bayes as nb
    from sklearn.naive_bayes import GaussianNB
    clf = nb.MultinomialNB()
    clf.fit(X_train, y_train)
    print clf.score(X_train, y_train)
    print clf.score(X_test, y_test)

if __name__ == '__main__':
    main()
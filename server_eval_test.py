import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

import codecs
import re
import common as cmn
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, accuracy_score


from scipy.ndimage import gaussian_filter


import warnings
warnings.filterwarnings("ignore")
fname = "./datasets/AFDpresidentutf8/corpus.tache1.learn.utf8"
alltxts,alllabs = cmn.load_pres(fname)

fname_test = "./test/presidents/corpus.tache1.learn.utf8"
alltxts_test= cmn.load_pres_test(fname_test)


"""{'reg__C': 100, 'reg__max_iter': 100, 'reg__penalty': 'l2', 'reg__tol': 0.0001, 'tfidf__binary': False, 'tfidf__lowercase': False, 'tfidf__max_df': 0.5, 'tfidf__max_features': 100000, 'tfidf__min_df': 2, 'tfidf__ngram_range': (1, 2)}"""
"""{'reg__C': 100, 'reg__max_iter': 100, 'reg__penalty': 'l2', 'reg__tol': 0.0001, 'tfidf__binary': False, 'tfidf__lowercase': False, 'tfidf__max_df': 0.5, 'tfidf__max_features': 100000, 'tfidf__min_df': 2, 'tfidf__ngram_range': (1, 2)}"""
# vectorizer_tf_idf_bigram = TfidfVectorizer(preprocessor=lambda x: (((cmn.majuscules_en_marqueurs((cmn.suppression_balises_html((x))))))),ngram_range=(1,2),binary=True,max_df=0.2,max_features= 200000)
vectorizer_tf_idf_bigram = TfidfVectorizer(preprocessor=lambda x: (((cmn.majuscules_en_marqueurs((cmn.suppression_balises_html((x))))))),max_df=0.5,min_df=2,max_features=100000,ngram_range=(1,2))

X_train = vectorizer.fit_transform(alltxts)
X_test = vectorizer.transform(alltxts_test)

t = 1e-4
C=10000
lr_clf = LogisticRegression(random_state=0, solver='liblinear',max_iter=10000, tol=t, C=C,penalty='l2')
lr_clf.fit(X_train, alllabs)
probabilites = lr_clf.predict_proba(X_test)
probabilites_metterand = probabilites[:,1]

probabilites_metterand_smoothed = gaussian_filter(probabilites_metterand,sigma=0.05)

print(probabilites_metterand_smoothed)



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
fname = "../datasets/AFDpresidentutf8/corpus.tache1.learn.utf8"
alltxts,alllabs = cmn.load_pres(fname)

fname_test = "../test/presidents/corpus.tache1.learn.utf8"
alltxts_test= cmn.load_pres_test(fname_test)


tfidf_params = {
    'max_df':  0.5,
    'min_df': 2,
    'ngram_range': (1, 2),
    'binary': True,
    'max_features': 100000,
}

reg_params = {
    # 'reg__C': [0.1, 1, 10, 100],
    'C': 10,
    'max_iter': 10000,
    'tol': 1e-4,
    'penalty': 'l2' ,
    'class_weight': {1: 1, -1: 5},
}

preprocessors = lambda x: ((cmn.suppression_chiffres(cmn.majuscules_en_marqueurs(cmn.suppression_ponctuation(cmn.suppression_balises_html((x)))))))

vectorizer_tf_idf_bigram = TfidfVectorizer(preprocessor=preprocessors,**tfidf_params)

X_train = vectorizer_tf_idf_bigram.fit_transform(alltxts)
X_test = vectorizer_tf_idf_bigram.transform(alltxts_test)


lr_clf = LogisticRegression(**reg_params)
lr_clf.fit(X_train, alllabs)
probabilites = lr_clf.predict_proba(X_test)
probabilites_metterand = probabilites[:,0]

probabilites_metterand_smoothed = gaussian_filter(probabilites_metterand,sigma=0.1)

print(probabilites)
print(probabilites.shape)
print(probabilites_metterand_smoothed)
np.savetxt("predictions/pred_presidents_8_bestF1.txt",probabilites_metterand_smoothed,fmt='%s')

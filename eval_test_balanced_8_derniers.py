
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
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
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
preprocessors = lambda x: ((cmn.suppression_chiffres(cmn.majuscules_en_marqueurs(cmn.suppression_ponctuation(cmn.suppression_balises_html((x)))))))


vectorizer_bigram = CountVectorizer(ngram_range=(2, 2),preprocessor=preprocessors)
corpus_dicours = alltxts
corpus_dicours_sparse_mat_bigram = vectorizer_bigram.fit_transform(corpus_dicours) # Output is a sparse matrix
print("Taille initiale du vocabulaire avec des bigrammes :",len(vectorizer_bigram.get_feature_names_out()))
frequence = np.array(corpus_dicours_sparse_mat_bigram.sum(axis=0))[0]
indices_tries = np.argsort(-frequence, kind='quicksort')

# Trier les sommes des colonnes en utilisant les indices triés
somme_colonnes_triees = [frequence[i] for i in indices_tries]
somme_colonnes_triees

bigrammes = vectorizer_bigram.get_feature_names_out()
bigrammes_100_plus_frequents = [bigrammes[i] for i in indices_tries[:100] ]

vectorizer_bigram = CountVectorizer(ngram_range=(1, 1),preprocessor=preprocessors)
corpus_dicours = alltxts
corpus_dicours_sparse_mat_bigram = vectorizer_bigram.fit_transform(corpus_dicours) # Output is a sparse matrix
print("Taille initiale du vocabulaire avec des bigrammes :",len(vectorizer_bigram.get_feature_names_out()))
frequence = np.array(corpus_dicours_sparse_mat_bigram.sum(axis=0))[0]
indices_tries = np.argsort(-frequence, kind='quicksort')

# Trier les sommes des colonnes en utilisant les indices triés
somme_colonnes_triees = [frequence[i] for i in indices_tries]
somme_colonnes_triees

bigrammes = vectorizer_bigram.get_feature_names_out()
unigrammes_100_plus_frequents = [bigrammes[i] for i in indices_tries[:100] ]



tfidf_params = {
    # 'max_df':  0.5,
    'min_df': 2,
    'ngram_range': (1, 2),
    'binary': True,
    'max_features': 90000,
    'stop_words':bigrammes_100_plus_frequents[:10]+unigrammes_100_plus_frequents[:5] ,

}

reg_params = {
    'C': 10,
    'max_iter': 1000,
    'tol': 1e-4,
    'penalty': 'l2' ,
    'class_weight': {1: 1, -1: 5},
}


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
np.savetxt("predictions/pred_presidents_10_les_derniers.txt",probabilites_metterand_smoothed,fmt='%s')

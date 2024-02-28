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

tfidf_params = {
    'max_df':  0.5,
    'min_df': 2,
    # 'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3) , (3,3)],
    'ngram_range': (1, 2),
    'binary': True,
    'lowercase': False,
    # 'tfidf__use_idf': [False, True],
    # 'tfidf__sublinear_tf': [False, True],
    'max_features': 100000,
}

reg_params = {
    # 'reg__C': [0.1, 1, 10, 100],
    'C': 100,
    'max_iter': 10000,
    'tol': 1e-4,
    'penalty': 'l2' ,
    'class_weight':'balanced',
}

preprocessors =lambda x: (((cmn.majuscules_en_marqueurs((cmn.suppression_balises_html((x)))))))
"""{'reg__C': 100, 'reg__max_iter': 100, 'reg__penalty': 'l2', 'reg__tol': 0.0001, 'tfidf__binary': False, 'tfidf__lowercase': False, 'tfidf__max_df': 0.5, 'tfidf__max_features': 100000, 'tfidf__min_df': 2, 'tfidf__ngram_range': (1, 2)}"""
"""{'reg__C': 100, 'reg__max_iter': 100, 'reg__penalty': 'l2', 'reg__tol': 0.0001, 'tfidf__binary': False, 'tfidf__lowercase': False, 'tfidf__max_df': 0.5, 'tfidf__max_features': 100000, 'tfidf__min_df': 2, 'tfidf__ngram_range': (1, 2)}"""
# vectorizer_tf_idf_bigram = TfidfVectorizer(preprocessor=lambda x: (((cmn.majuscules_en_marqueurs((cmn.suppression_balises_html((x))))))),ngram_range=(1,2),binary=True,max_df=0.2,max_features= 200000)
vectorizer_tf_idf_bigram = TfidfVectorizer(preprocessor=preprocessors,**tfidf_params)

X_train = vectorizer_tf_idf_bigram.fit_transform(alltxts)
X_test = vectorizer_tf_idf_bigram.transform(alltxts_test)


lr_clf = LogisticRegression(**reg_params)
lr_clf.fit(X_train, alllabs)
probabilites = lr_clf.predict_proba(X_test)
probabilites_metterand = probabilites[:,0]

probabilites_metterand_smoothed = gaussian_filter(probabilites_metterand,sigma=0.05)

print(probabilites)
print(probabilites.shape)
print(probabilites_metterand_smoothed)
np.savetxt("predictions/pred_presidents_7_bis.txt",probabilites_metterand_smoothed,fmt='%s')








# # # import numpy as np
# # # from sklearn.utils import resample
# # # from sklearn.linear_model import LogisticRegression
# # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # from scipy.ndimage import gaussian_filter
# # # import common as cmn

# # # # Chargement des données
# # # fname = "./datasets/AFDpresidentutf8/corpus.tache1.learn.utf8"
# # # alltxts, alllabs = cmn.load_pres(fname)

# # # fname_test = "./test/presidents/corpus.tache1.learn.utf8"
# # # alltxts_test = cmn.load_pres_test(fname_test)

# # # # Paramètres du TfidfVectorizer
# # # tfidf_params = {
# # #     'max_df': 0.2,
# # #     'ngram_range': (1, 3),
# # #     'binary': True,
# # #     'max_features': 100000,
# # # }

# # # # Paramètres de la régression logistique
# # # reg_params = {
# # #     'C': 10,
# # #     'max_iter': 10000,
# # #     'tol': 1e-4,
# # #     'penalty': 'l2',
# # #     'class_weight': 'balanced',
# # # }

# # # # Fonction de prétraitement
# # # preprocessors = lambda x: ((cmn.majuscules_en_marqueurs(cmn.suppression_ponctuation(cmn.suppression_balises_html(x)))))

# # # # Initialisation du TfidfVectorizer
# # # vectorizer_tf_idf_bigram = TfidfVectorizer(preprocessor=preprocessors, **tfidf_params)

# # # # Extraction des caractéristiques des données d'apprentissage et de test
# # # X_train = vectorizer_tf_idf_bigram.fit_transform(alltxts)
# # # X_test = vectorizer_tf_idf_bigram.transform(alltxts_test)

# # # # Création du modèle de régression logistique
# # # lr_clf = LogisticRegression(**reg_params)

# # # # Balancement des classes par undersampling
# # # minority_class = -1  # Classe minoritaire
# # # majority_class = 1   # Classe majoritaire

# # # # Indices des échantillons de chaque classe
# # # minority_indices = np.where(np.array(alllabs) == minority_class)[0]
# # # majority_indices = np.where(np.array(alllabs) == majority_class)[0]

# # # # Nombre d'échantillons dans chaque classe
# # # minority_class_size = len(minority_indices)
# # # majority_class_size = len(majority_indices)

# # # # Sous-échantillonnage aléatoire de la classe majoritaire pour égaler la taille de la classe minoritaire
# # # undersampled_majority_indices = resample(majority_indices, replace=False, n_samples=minority_class_size, random_state=42)

# # # # Concaténation des indices des deux classes
# # # undersampled_indices = np.concatenate([undersampled_majority_indices, minority_indices])

# # # # Utilisation des indices sous-échantillonnés pour extraire les données équilibrées
# # # X_train_balanced = X_train[undersampled_indices]
# # # y_train_balanced = np.array(alllabs)[undersampled_indices]

# # # # Entraînement du modèle sur les données équilibrées
# # # lr_clf.fit(X_train_balanced, y_train_balanced)

# # # # Prédiction des probabilités sur les données de test
# # # probabilities = lr_clf.predict_proba(X_test)
# # # probabilities_mitterrand = probabilities[:, 0]

# # # # Lissage des probabilités
# # # probabilities_mitterrand_smoothed = gaussian_filter(probabilities_mitterrand, sigma=0.05)

# # # # Affichage des résultats
# # # print(probabilities)
# # # print(probabilities.shape)
# # # print(probabilities_mitterrand_smoothed)

# # # # Enregistrement des prédictions dans un fichier
# # # np.savetxt("predictions/pred_presidents_8.txt", probabilities_mitterrand_smoothed, fmt='%s')
# # import numpy as np

# # import matplotlib.pyplot as plt
# # from collections import Counter
# # import pandas as pd

# # import codecs
# # import re
# # import common as cmn
# # from sklearn.model_selection import GridSearchCV
# # from sklearn.pipeline import Pipeline
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from nltk.tokenize import word_tokenize
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import make_scorer, f1_score, roc_auc_score, accuracy_score


# # from scipy.ndimage import gaussian_filter


# # import warnings
# # warnings.filterwarnings("ignore")
# # fname = "./datasets/AFDpresidentutf8/corpus.tache1.learn.utf8"
# # alltxts,alllabs = cmn.load_pres(fname)

# # fname_test = "./test/presidents/corpus.tache1.learn.utf8"
# # alltxts_test= cmn.load_pres_test(fname_test)

# # tfidf_params = {
# #     'max_df':  0.2,
# #     'min_df': 2,
# #     'ngram_range': (1, 3),
# #     'binary': True,
# #     'lowercase': False,
# #     'max_features': 100000,
# # }

# # reg_params = {
# #     'C': 10,
# #     'max_iter': 10000,
# #     'tol': 1e-4,
# #     'penalty': 'l2' ,
# # }

# # preprocessors = lambda x: ((cmn.majuscules_en_marqueurs(cmn.suppression_ponctuation(cmn.suppression_balises_html(x)))))
# # """{'reg__C': 100, 'reg__max_iter': 100, 'reg__penalty': 'l2', 'reg__tol': 0.0001, 'tfidf__binary': False, 'tfidf__lowercase': False, 'tfidf__max_df': 0.5, 'tfidf__max_features': 100000, 'tfidf__min_df': 2, 'tfidf__ngram_range': (1, 2)}"""
# # """{'reg__C': 100, 'reg__max_iter': 100, 'reg__penalty': 'l2', 'reg__tol': 0.0001, 'tfidf__binary': False, 'tfidf__lowercase': False, 'tfidf__max_df': 0.5, 'tfidf__max_features': 100000, 'tfidf__min_df': 2, 'tfidf__ngram_range': (1, 2)}"""
# # # vectorizer_tf_idf_bigram = TfidfVectorizer(preprocessor=lambda x: (((cmn.majuscules_en_marqueurs((cmn.suppression_balises_html((x))))))),ngram_range=(1,2),binary=True,max_df=0.2,max_features= 200000)
# # vectorizer_tf_idf_bigram = TfidfVectorizer(preprocessor=preprocessors,**tfidf_params)

# # X_train = vectorizer_tf_idf_bigram.fit_transform(alltxts)
# # X_test = vectorizer_tf_idf_bigram.transform(alltxts_test)


# # lr_clf = LogisticRegression(**reg_params)
# # lr_clf.fit(X_train, alllabs)
# # probabilites = lr_clf.predict_proba(X_test)
# # probabilites_metterand = probabilites[:,0]

# # probabilites_metterand_smoothed = gaussian_filter(probabilites_metterand,sigma=0.05)

# # print(probabilites)
# # print(probabilites.shape)
# # print(probabilites_metterand_smoothed)
# # np.savetxt("predictions/pred_presidents_8_bis.txt",probabilites_metterand_smoothed,fmt='%s')
# import numpy as np
# from sklearn.utils import resample
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import TfidfVectorizer
# from scipy.ndimage import gaussian_filter
# import common as cmn

# # Chargement des données
# fname = "./datasets/AFDpresidentutf8/corpus.tache1.learn.utf8"
# alltxts, alllabs = cmn.load_pres(fname)

# fname_test = "./test/presidents/corpus.tache1.learn.utf8"
# alltxts_test = cmn.load_pres_test(fname_test)


# tfidf_params = {
#     'max_df':  0.5,
#     'min_df': 2,
#     # 'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3) , (3,3)],
#     'ngram_range': (1, 2),
#     'binary': True,
#     'lowercase': False,
#     # 'tfidf__use_idf': [False, True],
#     # 'tfidf__sublinear_tf': [False, True],
#     'max_features': 100000,
# }

# reg_params = {
#     # 'reg__C': [0.1, 1, 10, 100],
#     'C': 100,
#     'max_iter': 10000,
#     'tol': 1e-4,
#     'penalty': 'l2' ,
#     'class_weight':'balanced',
# }
# # Fonction de prétraitement
# preprocessors =lambda x: (((cmn.majuscules_en_marqueurs((cmn.suppression_balises_html((x)))))))

# # Initialisation du TfidfVectorizer
# vectorizer_tf_idf_bigram = TfidfVectorizer(preprocessor=preprocessors, **tfidf_params)

# # Extraction des caractéristiques des données d'apprentissage et de test
# X_train = vectorizer_tf_idf_bigram.fit_transform(alltxts)
# X_test = vectorizer_tf_idf_bigram.transform(alltxts_test)

# # Création du modèle de régression logistique
# lr_clf = LogisticRegression(**reg_params)

# # Balancement des classes par undersampling
# minority_class = -1  # Classe minoritaire
# majority_class = 1   # Classe majoritaire

# # Indices des échantillons de chaque classe
# minority_indices = np.where(np.array(alllabs) == minority_class)[0]
# majority_indices = np.where(np.array(alllabs) == majority_class)[0]

# # Nombre d'échantillons dans chaque classe
# minority_class_size = len(minority_indices)
# majority_class_size = len(majority_indices)

# # Sous-échantillonnage aléatoire de la classe majoritaire pour égaler la taille de la classe minoritaire
# if majority_class_size >= minority_class_size:
#     undersampled_majority_indices = resample(majority_indices, replace=False, n_samples=minority_class_size, random_state=42)

#     # Concaténation des indices des deux classes
#     undersampled_indices = np.concatenate([undersampled_majority_indices, minority_indices])

#     # Utilisation des indices sous-échantillonnés pour extraire les données équilibrées
#     X_train_balanced = X_train[undersampled_indices]
#     y_train_balanced = np.array(alllabs)[undersampled_indices]

#     # Entraînement du modèle sur les données équilibrées
#     lr_clf.fit(X_train_balanced, y_train_balanced)

#     # Prédiction des probabilités sur les données de test
#     probabilities = lr_clf.predict_proba(X_test)
#     probabilities_mitterrand = probabilities[:, 0]

#     # Lissage des probabilités
#     probabilities_mitterrand_smoothed = gaussian_filter(probabilities_mitterrand, sigma=0.05)

#     # Affichage des résultats
#     print(probabilities)
#     print(probabilities.shape)
#     print(probabilities_mitterrand_smoothed)

#     # Enregistrement des prédictions dans un fichier
#     np.savetxt("predictions/pred_presidents_8_quatro.txt", probabilities_mitterrand_smoothed, fmt='%s')
# else:
#     print("La classe majoritaire a moins d'échantillons que la classe minoritaire. Sous-échantillonnage évité.")

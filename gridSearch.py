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

print(len(alltxts),len(alllabs))
print(alltxts[0])
print(alllabs[0])
print(alltxts[-1])
print(alllabs[-1])

"""{'reg__C': 100, 'reg__max_iter': 100, 'reg__penalty': 'l2', 'reg__tol': 0.0001, 'tfidf__binary': False, 'tfidf__lowercase': False, 'tfidf__max_df': 0.5, 'tfidf__max_features': 100000, 'tfidf__min_df': 2, 'tfidf__ngram_range': (1, 2)}"""
# vectorizer_tf_idf_bigram = TfidfVectorizer(preprocessor=lambda x: (((cmn.majuscules_en_marqueurs((cmn.suppression_balises_html((x))))))),ngram_range=(1,2),binary=True,max_df=0.2,max_features= 200000)
vectorizer_tf_idf_bigram = TfidfVectorizer(preprocessor=lambda x: (((cmn.majuscules_en_marqueurs((cmn.suppression_balises_html((x))))))))

class_weights = {
    1: 1.0,  # Class 1, the majority class, gets weight 1.0 (default weight)
    -1: 5.0  # Class -1, the minority class, gets weight 5.0
}

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(preprocessor=lambda x: (((cmn.majuscules_en_marqueurs((cmn.suppression_balises_html((x))))))))),
    ('reg', LogisticRegression())
])

parameters = {
    'tfidf__max_df': [ 0.5],
    'tfidf__min_df': [2],
    # 'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3) , (3,3)],
    'tfidf__ngram_range': [ (1, 2)],
    'tfidf__binary': [True],
    # 'tfidf__use_idf': [False, True],
    # 'tfidf__sublinear_tf': [False, True],
    'tfidf__max_features': [40000],
    'reg__class_weight': ['balanced'],  
    'reg__C': [100000],
    # 'reg__C': [100],
    'reg__max_iter': [10000],
    'reg__tol': [1e-4],
    'reg__penalty': [ 'l2' ]

}

scoring = {
    'f1_score': make_scorer(f1_score),
    'roc_auc': make_scorer(roc_auc_score),
    'accuracy': make_scorer(accuracy_score)
}

[X_all_train, X_all_test, Y_train, y_test]  = train_test_split(alltxts, alllabs, test_size=0.2, random_state=10, shuffle=True)


grid_search = GridSearchCV(pipeline, parameters,scoring=scoring, refit='f1_score', cv=5, n_jobs=-1, verbose=1)


grid_search.fit(X_all_train, Y_train)


print("Meilleurs paramètres trouvés:")
print(grid_search.best_params_)
#_____________

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_all_test)

smoothed_pred = gaussian_filter(y_pred,sigma=0.1)

# Calcul des métriques de performance après le lissage
f1 = f1_score(y_test, smoothed_pred, average='micro')  # or 'macro' or 'weighted'
roc_auc = roc_auc_score(y_test, smoothed_pred)
accuracy = accuracy_score(y_test, (smoothed_pred > 0.5).astype(int))

print("Performance après lissage:")
print("F1 Score:", f1)
print("AUC:", roc_auc)
print("Accuracy:", accuracy)




#_____________
# print("Scores:")
# print("F1 Score:", grid_search.cv_results_['mean_test_f1_score'])
# print("AUC:", grid_search.cv_results_['mean_test_roc_auc'])
# print("Accuracy:", grid_search.cv_results_['mean_test_accuracy'])
if __name__ == '__main__':
    # Execute when the module is not initialized from an import statement.
    print("hey")




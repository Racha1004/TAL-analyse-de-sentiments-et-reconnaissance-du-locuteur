
import numpy as np

import common_bis as cmn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler


from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d


import warnings
warnings.filterwarnings("ignore")



fname = "../datasets/AFDpresidentutf8/corpus.tache1.learn.utf8"
alltxts,alllabs = cmn.load_pres(fname)

fname_test = "../test/presidents/corpus.tache1.learn.utf8"
alltxts_test= cmn.load_pres_test(fname_test)

from unidecode import unidecode

def supprimer_accents(texte):
    return unidecode(texte)

def preprocessors_fct (x):
    x_1 = (cmn.majuscules_en_marqueurs(cmn.suppression_ponctuation((cmn.suppression_balises_html(x)))))
    x_2 =(cmn.lemmatization(cmn.remove_stopwords(cmn.suppression_chiffres(x_1))))
    return x_2
alltxts_cleand = [preprocessors_fct (x) for x in alltxts]
att_test_cleaned = [preprocessors_fct (x) for x in alltxts_test]


tf_idf_vectorizer =TfidfVectorizer(min_df=5, ngram_range=(1, 3),binary=True,max_df=0.5)

X_train = tf_idf_vectorizer.fit_transform(alltxts_cleand)
X_test = tf_idf_vectorizer.transform(att_test_cleaned)


t = 1e-8
C=10
lr_clf = LogisticRegression(random_state=0, solver='liblinear',max_iter=1000, tol=t, C=C,class_weight='balanced')
lr_clf.fit(X_train, alllabs)


probabilites = lr_clf.predict_proba(X_test)
probabilites_metterand = probabilites[:,0]

probabilites_metterand_smoothed = gaussian_filter(probabilites_metterand,sigma=2.)

print(probabilites)
print(probabilites.shape)
print(probabilites_metterand_smoothed)
np.savetxt("predictions/pred_presidents_9_29_final_7.txt",probabilites_metterand_smoothed,fmt='%s')

# => best version 14,37 le final 7















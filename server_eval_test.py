
import numpy as np

import common as cmn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler


from scipy.ndimage import gaussian_filter


import warnings
warnings.filterwarnings("ignore")
fname = "./datasets/AFDpresidentutf8/corpus.tache1.learn.utf8"
alltxts,alllabs = cmn.load_pres(fname)

fname_test = "./test/presidents/corpus.tache1.learn.utf8"
alltxts_test= cmn.load_pres_test(fname_test)


tfidf_params = {
    'max_df':  0.5,
    'ngram_range': (1, 2),
    'binary': True,
}

reg_params = {
    # 'reg__C': [0.1, 1, 10, 100],
    'C': 10,
    'max_iter': 10000,
    'tol': 1e-4,
    'penalty': 'l2' ,
}

preprocessors = lambda x: (((cmn.majuscules_en_marqueurs(cmn.suppression_ponctuation(cmn.suppression_balises_html((x)))))))

# vectorizer_tf_idf_bigram = TfidfVectorizer(preprocessor=lambda x: (((cmn.majuscules_en_marqueurs((cmn.suppression_balises_html((x))))))),ngram_range=(1,2),binary=True,max_df=0.2,max_features= 200000)
vectorizer_tf_idf_bigram = TfidfVectorizer(preprocessor=preprocessors,**tfidf_params)

X_train = vectorizer_tf_idf_bigram.fit_transform(alltxts)
X_test = vectorizer_tf_idf_bigram.transform(alltxts_test)

oversampler = RandomOverSampler(random_state=42)
X_train_resampled, Y_train_resampled = oversampler.fit_resample(X_train, alllabs)



lr_clf = LogisticRegression(**reg_params)
lr_clf.fit(X_train_resampled, Y_train_resampled)
probabilites = lr_clf.predict_proba(X_test)
probabilites_metterand = probabilites[:,0]

probabilites_metterand_smoothed = gaussian_filter(probabilites_metterand,sigma=0.1)

print(probabilites)
print(probabilites.shape)
print(probabilites_metterand_smoothed)
np.savetxt("predictions/pred_presidents_9_28_02_oversampl_4_final.txt",probabilites_metterand_smoothed,fmt='%s')

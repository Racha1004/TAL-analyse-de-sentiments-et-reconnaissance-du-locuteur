import numpy as np
import matplotlib.pyplot as plt

import codecs
import re
import os.path


import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


#-------------------FONCTIONS POUR LE CHARGEMENT DE DONNÉES------------------#
# Données reconnaissance du locuteur (Chirac/Mitterrand)
def load_pres(fname):
    alltxts = []
    alllabs = []
    s=codecs.open(fname, 'r','utf-8') # pour régler le codage
    while True:
        txt = s.readline()
        if(len(txt))<5:
            break
        print("tx1t",txt)
        lab = re.sub(r"<[0-9]*:[0-9]*:(.)>.*","\\1",txt)
        print("lab",lab)
        txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)","\\1",txt)
        print("txt2",txt)
        if lab.count('M') >0:
            alllabs.append(-1)
        else: 
            alllabs.append(1)
        alltxts.append(txt)
    return alltxts,alllabs


# Données classification de sentiments (films)
def load_movies(path2data): # 1 classe par répertoire
    alltxts = [] # init vide
    labs = []
    cpt = 0
    for cl in os.listdir(path2data): # parcours des fichiers d'un répertoire
        for f in os.listdir(path2data+cl):
            txt = open(path2data+cl+'/'+f).read()
            alltxts.append(txt)
            labs.append(cpt)
        cpt+=1 # chg répertoire = cht classe
        
    return alltxts,labs



#-------------------------FONCTIONS POUR LE PREPROCESSING--------------------#

def remove_ponctuation(sentence):
    """Supprimer la ponctuation"""
    return re.sub(r'[^\w\s]', '', sentence)


def tokenize_sentence(sentence):
    """Conversion en des listes de tokens
    download punkt et wordnet avec nltk.donwload"""
    return word_tokenize(sentence)



def remove_stopwords():
    """Supprimer les stop words 
    On peut etre cette liste en faisant stop_words.append("Alasca")"""
    stopwords.words('french')


from nltk.stem.snowball import FrenchStemmer
def stemming(sentence):
    fr_stemmer = FrenchStemmer()
    tokens = tokenize_sentence(sentence)
    sentence_stemmed = []
    for word in tokens:
        sentence_stemmed.append(fr_stemmer.stem(word))
    return " ".join(sentence_stemmed)

from nltk.stem import WordNetLemmatizer
def lemmatization(sentence):
    lemmatizer = WordNetLemmatizer()
    tokens = tokenize_sentence(sentence)
    sentence_lemmatized = []
    for word in tokens:
        sentence_lemmatized.append(lemmatizer.lemmatize(word))
    return " ".join(sentence_lemmatized)







print(tokenize_sentence("BOnjour le monde , merci"))
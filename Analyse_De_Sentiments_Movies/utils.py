import numpy as np
import matplotlib.pyplot as plt

import codecs
import re
import os.path

import string
import re
import unicodedata
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.stem.snowball import EnglishStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

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

def load_movies_test(fname):
    return open(fname).readlines()

#-------------------------FONCTIONS POUR LE PREPROCESSING--------------------#
def remove_ponctuation(sentence):
    punc = string.punctuation  
    punc += '\n\r\t'
    sentence = sentence.translate(str.maketrans(punc, ' ' * len(punc)))
    sentence = re.sub('( )+', ' ', sentence)  # Supprimer les espaces en trop
    return sentence

def remove_numbers(sentence):
    return re.sub('[0-9]+', '', sentence)

def suppression_balises_html(text):
    motif = re.compile(r'<[^>]+>')
    return re.sub(motif, '', text)

def remove_names_En(sentence):
    # Tokenisation des mots dans la phrase
    tokens = nltk.word_tokenize(sentence)
    # Obtenir les étiquettes POS (Part of Speech) pour chaque mot
    pos_tags = nltk.pos_tag(tokens)
    # Filtrer les mots qui ne sont pas des noms propres
    tokens_filtered = [word for (word, pos) in pos_tags if pos != 'NNP']
    # Reconstruire la phrase avec les mots filtrés
    sentence_filtered = ' '.join(tokens_filtered)
    return sentence_filtered

def stemming_En(sentence):
    stemmer = EnglishStemmer()
    tokens = nltk.word_tokenize(sentence)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    stemmed_text = " ".join(stemmed_tokens)
    return stemmed_text

def lemmatization(sentence):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(sentence)
    sentence_lemmatized = []
    for word in tokens:
        sentence_lemmatized.append(lemmatizer.lemmatize(word))
    return " ".join(sentence_lemmatized)

def remove_stopwords_En(sentence):
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(sentence)
    filtered_sentence = [word.lower()  for word in tokens if word.lower() not in stop_words]
    return " ".join(filtered_sentence)

def extraire_debut(text):
    return re.split(r'[.?!]',text)[0]

def extraire_fin(text):
    if text.endswith('\n'): 
        text = text[:-1]
    return re.split(r'[.!?]', text)[-2]
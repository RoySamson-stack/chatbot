import random 
import json 
import pickle 
import numpy as np
import tensorflow as tf

import nltk 
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from nltk.tokenize import word_tokenize


lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')
intents = json.loads(open('intents.json').read())

words =[]
classes = []
documents = []
ignore_letters = ['?', "!", ".", ","]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])



words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

print(words)

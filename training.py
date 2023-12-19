import random 
import json 
import pickle 
import numpy as np

import nltk 
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizer import SGD


lemmatizer = WordNetLemmatizer

intents = json.loads(open('intents.json').read())

words =[]
classes = []
documents = []
ignore_letters = ['?', "!", ".", ","]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.tokenize(pattern)




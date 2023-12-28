import random
import json
import pickle
import numpy as np
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import tensorflow as tf

nltk.download('punkt')
nltk.download('wordnet')

intents_file = 'intents.json'
intents = json.loads(open(intents_file).read())

def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    patterns = [element.text for element in soup.select('.pattern-class')]
    responses = [element.text for element in soup.select('.response-class')]
    
    return {'patterns': patterns, 'responses': responses}

website_url = 'https://en.wikipedia.org/wiki/Category:English_phrases'
scraped_data = scrape_website(website_url)

new_intent = {
    "tag": "new_intent",
    "patterns": scraped_data['patterns'],
    "responses": scraped_data['responses']
}

intents['intents'].append(new_intent)

with open(intents_file, 'w') as json_file:
    json.dump(intents, json_file, indent=4)

lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words]

training_x = []
training_y = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training_x.append(bag)
    training_y.append(output_row)

combined_data = list(zip(training_x, training_y))
random.shuffle(combined_data)
training_x[:], training_y[:] = zip(*combined_data)

train_x = np.array(training_x)
train_y = np.array(training_y)

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

sgd = tf.keras.optimizers.legacy.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

model.save('chatbot_model.h5', hist)

print("Done")

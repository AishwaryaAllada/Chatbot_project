#load all the required libraries
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
nltk.download('wordnet')


words=[] #to store the list of all words
classes = [] # to store the list of all the categories("tag")
documents = [] #list to store the tokenized words of a sentence along wiwth the tag
ignore_words = ['?', '!']


# data_file = open('intents.json')
# data = json.load(data_file)
# for i in data["intents"]:
# 	print(i)

data_file = open('intents.json').read() #reading the json file having patterns and reponses
intents = json.loads(data_file) #load the file for further process
# print(intents)

for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize each word from each sentence
        w = nltk.word_tokenize(pattern)
        words.extend(w) # add the list of words to the "words" list
        #add documents in the corpus
        documents.append((w, intent['tag']))#tokenized sentence along with its tag is stored
         # add to our classes list-- printing only unique values of classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])



print("words are", words)
print()
print()
print()
print("documents are", documents)
print()
print()
print()
print("Classes are", classes)


#now after tthe iterations, "words" contains all the words in the vocab
# lemmatize, lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
# sort words -- alphabetical order
words = sorted(list(set(words)))
print()
print()
print()
print("sorted list of words", words)
# sort classes -- alphabetical order
classes = sorted(list(set(classes)))
print()
print()
print()
print("sorted list of classes",classes)
# documents = combination between patterns and intents
print (len(documents), "no of documents")
print()
print()
print()
# classes = intents
print (len(classes), "no of classes", classes)
print()
print()
print()
# words = all words, vocabulary
print (len(words), "no of unique lemmatized words", words)


#store the vocab in a pickel file
pickle.dump(words,open('words.pkl','wb'))

#store the classes in pickel file
pickle.dump(classes,open('classes.pkl','wb'))



#######################################
#Create training and testing data

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)


# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

# print()
# print("///////////////////")
# print(bag)
# print(len(bag))

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])


# print(training)

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

#train the model

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import string
import random

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer

import os
os.environ['TF_USE_LEGACY_KERAS'] = 'True'
# In[2]:


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout


# In[9]:


import json
import requests
# import pandas as pd 

response = requests.get("https://api.avila.aesirasdf.com/api/intentall")

if response.status_code == 200:
    # Get the data from the response
    data1 = response.json()  # For JSON response
    # print(data1['intents'])
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")


# In[42]:


# Assuming data1 is already defined and contains the "intents" key
# df = pd.DataFrame(data1["intents"])

# Select the relevant columns
# selected_columns = df[['tag', 'user_pattern', 'bot_response']]

# Filter out rows where 'user_pattern' or 'bot_response' are empty arrays
# filtered_columns = selected_columns[
#     selected_columns.apply(lambda row: len(row['user_pattern']) > 0 and len(row['bot_response']) > 0, axis=1)
# ]

# filtered_columns


# In[43]:


# # Opening JSON file
# f = open('C:\\Users\\User\\Desktop\\Research\\Chatbot\\intents.json', encoding="utf8")

# # returns JSON object as 
# # a dictionary
# data = json.load(f)

# # Iterating through the json
# # list
# for i in data['intents']:
#     print (i)
    
# # Closing file
# f.close()


# In[41]:


# data_first = data["intents"]
# data_first 


# In[45]:


data_filtered = [intent for intent in data1['intents'] if intent['user_pattern'] and intent['bot_response']]
data_filtered


# In[47]:


# this code filters the data showing only tags, patterns, responses and context, and not allowing the empty arrays to be shown.

data_filtered = [
    {
        'tag': intent['tag'],
        'patterns': intent['user_pattern'],
        'responses': intent['bot_response'],
        'context': intent['context']
    }
    for intent in data1['intents']
    if intent['user_pattern'] and intent['bot_response']
]

data_filtered


# In[52]:


## Creating x_data and y_data

words = [] # Bow Model / Vocabulary For Patterns
classes = [] # Bow Model / Vocabulary for Tags

x_data = [] # For storing each pattern
y_data = [] # For storing tag corresponding to each patternin x_data

# Iterating over all of the intent
for intent in data_filtered:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern) # Tokenize Each Pattern
        words.extend(tokens) # and append tokens to words
        x_data.append(pattern) # appending pattern to list x_data
        y_data.append(intent["tag"]) # appending the associated tag to each pattern
        
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

# Initilialize lemmatizer to get stem of words
lemmatizer = WordNetLemmatizer()

# lemmatize all the words in the vocabulary and convert them to lowercase
# if the words don't appear in punctuation

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]

# Sorting the vocabulary and classes in alphabetical order and taking the # set to ensure no duplicates occur

words = sorted(set(words))
classes = sorted(set(classes))


# In[51]:


x_data,y_data


# In[53]:


x_data,y_data


# ### Text to numbers

# In[54]:


training = []
output_empty= [0] * len(classes)

#Creating the bag of words model

for idx, doc in enumerate(x_data):
    bow =[]
    text = lemmatizer.lemmatize(doc.lower())
    
    for word in words:
        bow.append(1) if word in text else bow.append(0)
    # Mark the index of classes that the current pattern is associated
    # to
    output_row = list(output_empty)
    output_row[classes.index(y_data[idx])] = 1
    
    # add the not encoded bow and associated classes to training
    training.append([bow, output_row])
#shuffle the data and convert it to an array

random.shuffle(training)
training = np.array(training, dtype=object)

# split the features and target labels
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))


# <!-- training[:, 0] -->

# <!-- training[:, 1] -->

# ### Neural Network Model

# In[55]:


model = Sequential()
model.add(Dense(128, input_shape = (len(train_x[0]), ), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation = "softmax"))
adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss = "categorical_crossentropy",
             optimizer = adam,
             metrics = ["accuracy"])

# print(model.summary())
history = model.fit(x=train_x, y=train_y, epochs=150, verbose=1)
model_accuracy = history.history['accuracy'][-1]
# model.fit(x=train_x, y=train_y, epochs=150, verbose = 1)


# ### Preprocessing the Input
# 

# In[56]:


def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1    
    return np.array(bow)


# In[57]:


def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]))[0] # Extracting possibilities
    thresh = 0.5
    
    y_pred = [[indx, res] for indx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True) # Sorting by values of probability in decreasing order
    
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]]) # contains labels for highest probability
    return return_list


# In[73]:


def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        result = "intents 0"
    else:
        tag = intents_list[0]
       #list_of_intents = intents_json["intents"]
        for i in intents_json:
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                break
    return result


# Let us quickly discuss these functions in more detail:
# 
# Clean_text(text): This function receives text (string) as an input and then tokenizes it using the nltk.word_tokenize(). Each token is then converted into its root form using a lemmatizer. The output is basically a list of words in their root form.
# 
# Bag_of_words(text, vocab): This function calls the above function, converts the text into an array using the bag-of-words model using the input vocabulary, and then returns the same array. 
# 
# Pred_class(text, vocab, labels): This function takes text, vocab, and labels as input and returns a list that contains a tag corresponding to the highest probability.
# 
# Get_response(intents_list, intents_json): This function takes in the tag returned by the previous function and uses it to randomly choose a response corresponding to the same tag in intent.json. And, if the intents_list is empty, that is when the probability does not cross the threshold, we will pass the string “Sorry! I don’t understand” as ChatBot’s response.

# #### Step-8: Calling the Relevant Functions and interacting with the ChatBot

# In[ ]:


# Interacting with the chatbot

#lOGIN SECTION!!

# print("Type 'stop' if you don't want to chat with ORCA chatbot")
# while True:
#     message = input("")
#     if message == 'stop':
#         break
#     intents = pred_class(message, words, classes)
#     result = get_response(intents, data_filtered)
#     print(result)


# <a href= "https://www.projectpro.io/article/python-chatbot-project-learn-to-build-a-chatbot-from-scratch/429"> Source </a>




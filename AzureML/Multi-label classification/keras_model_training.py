import numpy as np
import argparse
import os
import glob
from array import array
import pandas as pd
import re

import matplotlib.pyplot as plt

import nltk
from nltk.stem import PorterStemmer

from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense
from keras.optimizers import adam
from keras.callbacks import Callback

import tensorflow as tf
import joblib


from azureml.core import Run

print("Keras version:", keras.__version__)
print("Tensorflow version:", tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument("--input-data", type=str, dest='training_dataset_id', help='training dataset')
parser.add_argument('--epoch-size', type=int, dest='epoch_size', default=100, help='epoch size for training')

args = parser.parse_args()

# start an Azure ML run
run = Run.get_context()

# Get the training dataset
print("Loading Data...")
data = run.input_datasets['training_data'].to_pandas_dataframe()

stemmer = PorterStemmer()
# Drop rows with empty description
data.dropna(inplace=True)
# NAC removal            
data['description'].replace(to_replace="[\(\*]NA[C]?\s*[\)\(]*C?", value=r"", regex=True, inplace=True)
# Stem, remnove special characters, and lowecase transformation
data['description'] = data['description'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z0-9]", " ", x).split()]).lower())

vectorizer = CountVectorizer(stop_words="english", analyzer='word', ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None)
v1=vectorizer.fit_transform(data['description'])


# Separate features and labels
X=v1.toarray()
y=data.iloc[:,1:]


# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=12345)

n_inputs, n_outputs = X.shape[1], y.shape[1]
nodes_number=200
n_epochs = args.epoch_size
#learning_rate = args.learning_rate

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep='\n')

# Build a simple MLP model
model = Sequential()
# a hidden layer
model.add(Dense(nodes_number, input_dim=n_inputs, kernel_initializer='he_normal', activation='relu'))
# output layer
model.add(Dense(n_outputs, activation='sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])




class LogRunMetrics(Callback):
    # callback at the end of every epoch
    def on_epoch_end(self, epoch, log):
        # log a value repeated which creates a list
        run.log('Loss', log['val_loss'])
        run.log('Accuracy', log['val_accuracy'])


history = model.fit(X_train, y_train, epochs=n_epochs, verbose=3, validation_data=(X_test, y_test), callbacks=[LogRunMetrics()])
score = model.evaluate(X_test, y_test, verbose=0)

# log a single value
run.log("Final test loss", score[0])
print('Test loss:', score[0])

run.log('Final test accuracy', score[1])
print('Test accuracy:', score[1])

plt.figure(figsize=(6, 3))
plt.title('NN with Keras MLP ({} epochs)'.format(n_epochs), fontsize=14)
plt.plot(history.history['val_accuracy'], 'b-', label='Accuracy', lw=4, alpha=0.5)
plt.plot(history.history['val_loss'], 'r--', label='Loss', lw=4, alpha=0.5)
plt.legend(fontsize=12)
plt.grid(True)

# log an image
run.log_image('Accuracy vs Loss', plot=plt)


# create a ./outputs/model folder in the compute target
# files saved in the "./outputs" folder are automatically uploaded into run history
os.makedirs('./outputs/model', exist_ok=True)

joblib.dump(vectorizer, './outputs/model/feature_vectorizer.joblib') 
joblib.dump(y.columns.tolist(), './outputs/model/tags.joblib')

# serialize NN architecture to JSON
model_json = model.to_json()
# save model JSON
with open('./outputs/model/model-1.json', 'w') as f:
    f.write(model_json)
# save model weights
model.save_weights('./outputs/model/model-1.h5')
print("model saved in ./outputs/model folder")


run.complete()


from pathlib import Path
from os.path import exists
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.utils as util
import tensorflow.keras.layers as lr
import tensorflow.keras.preprocessing as prp
import tensorflow_hub as hub

class TensorFlow:
    def __init__(self, trainingPath, validationPath, testingPath):
        desired_width = 320
        pd.set_option('display.width', desired_width)
        pd.set_option('display.max_columns', 5)

        # Initialise class variables
        self.mTrainingPath = trainingPath
        self.mValidationPath = validationPath
        self.mTestingPath = testingPath

        # Load datasets
        self.loadDatasets()

        return

    def loadDatasets(self):
        self.mTrainingData = pd.read_csv(self.mTrainingPath, usecols=['title', 'text'], encoding='utf8').to_numpy(dtype=None)
        self.mTrainingLabels = pd.read_csv(self.mTrainingPath, usecols=['label'], encoding='utf8').to_numpy(dtype=None)
        self.createWordIndex()


        return

    def loadModel(self):
        # Check for existence of saved model, and if so load the model
        if exists(Path("Tensorflow.h5")):
            self.mModel = ks.models.load_model("Tensorflow.h5")
            return

        # We want to evaluate 2 columns; the title and text columns. So we have 2 inputs for each column
        input1 = lr.Input(shape=(200,))
        input2 = lr.Input(shape=(200,))
        # Combine the columns
        merged = lr.Concatenate(axis=1)([input1, input2])
        # Embed the encoded text (helps the lstm)
        embedding = lr.Embedding(len(self.mToken.word_index)+1, 32, input_length=200)(merged)
        lstm = lr.LSTM(32)(embedding)
        output = lr.Dense(1, activation="sigmoid")(lstm)
        # Create the functional model
        self.mModel = ks.Model(inputs=[input1, input2], outputs=output)

        self.mModel.summary()

        print(self.mModel.input_shape)

        self.mModel.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['acc'])

        return

    def fitModel(self):
        history = self.mModel.fit([self.mTrainingData[0], self.mTrainingData[1]], self.mTrainingLabels, epochs=10, validation_split=0.2)
        return

    def createWordIndex(self):
        # Create tokenizer and create a "bag of words" like index
        token = prp.text.Tokenizer()
        token.fit_on_texts(self.mTrainingData.T[0] + self.mTrainingData.T[1])

        # Takes a string and returns a list of padded integers representing the encoded string
        def textToIndex(text):
            tokens = [token.word_index[word] if word in token.word_index else 0 for word in prp.text.text_to_word_sequence(text)]
            return list(prp.sequence.pad_sequences([tokens], 200)[0].reshape(-1))

        # Create temporary list
        temp = []
        # Iterate through transposed training data and encode each string as list of integers
        for rowInd in range(len(self.mTrainingData.T)):
            temp.append([textToIndex(x) for x in self.mTrainingData.T[rowInd]])
        # Create new numpy array with encoded strings
        self.mTrainingData = np.array(temp, dtype=np.int)
        # Make sure to save token for later use
        self.mToken = token
        return

    mTrainingPath = None
    mValidationPath = None
    mTestingPath = None
    mTrainingData = None
    mTrainingLabels = None
    mValidationData = None
    mValidationLabels = None
    mTestingData = None
    mTestingLabels = None
    mWordIndex = None
    mModel = None

    mToken = None
    mMaxLen = None
    mBatchSize = 32
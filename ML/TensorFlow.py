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

        # Create word index for encoding strings
        self.createWordIndex()

        # Encode the strings
        self.mTrainingData = self.encodeTable(self.mTrainingData, self.mToken)
        self.mValidationData = self.encodeTable(self.mValidationData, self.mToken)
        self.mTestingData = self.encodeTable(self.mTestingData, self.mToken)

        return

    def loadDatasets(self):
        # Get the training set
        self.mTrainingData = pd.read_csv(self.mTrainingPath, usecols=['title', 'text'],
                                         encoding='utf8').to_numpy(dtype=None).T
        self.mTrainingLabels = pd.read_csv(self.mTrainingPath, usecols=['label'],
                                           encoding='utf8').to_numpy(dtype=None)
        # Get the validation set
        self.mValidationData = pd.read_csv(self.mValidationPath, usecols=['title', 'text'],
                                         encoding='utf8').to_numpy(dtype=None).T
        self.mValidationLabels = pd.read_csv(self.mValidationPath, usecols=['label'],
                                           encoding='utf8').to_numpy(dtype=None)
        # Get the testing set
        self.mTestingData = pd.read_csv(self.mTestingPath, usecols=['title', 'text'],
                                         encoding='utf8').to_numpy(dtype=None).T
        self.mTestingLabels = pd.read_csv(self.mTestingPath, usecols=['label'],
                                           encoding='utf8').to_numpy(dtype=None)
        return

    def createWordIndex(self):
        # Create tokenizer and create a "bag of words" like index
        token = prp.text.Tokenizer()
        token.fit_on_texts(self.mTrainingData[0] + self.mTrainingData[1])
        # Make sure to save token for later use
        self.mToken = token
        return

    def encodeTable(self, table, token):
        # Takes a string and returns a list of padded integers representing the encoded string
        def textToIndex(text):
            tokens = [token.word_index[word] if word in token.word_index else 0 for word in
                      prp.text.text_to_word_sequence(text)]
            return list(prp.sequence.pad_sequences([tokens], 200)[0].reshape(-1))

        # Create temporary list
        temp = []
        # Iterate through transposed training data and encode each string as list of integers
        for rowInd in range(len(table)):
            temp.append([textToIndex(x) for x in table[rowInd]])
        # Create new numpy array with encoded strings
        return np.array(temp, dtype=np.int)

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
        embedding = lr.Embedding(len(self.mToken.word_index)+1, 64, input_length=200)(merged)
        lstm = lr.LSTM(64)(embedding)
        output = lr.Dense(1, activation="sigmoid")(lstm)
        # Create the functional model
        self.mModel = ks.Model(inputs=[input1, input2], outputs=output)

        self.mModel.summary()

        print(self.mModel.input_shape)

        self.mModel.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['acc'])
        return

    def fitModel(self):
        history = self.mModel.fit([self.mTrainingData[0], self.mTrainingData[1]], self.mTrainingLabels, epochs=3,
                                  batch_size=64, shuffle=True,
                                  validation_data=([self.mValidationData[0], self.mValidationData[1]], self.mValidationLabels))
        return

    def evaluate(self):
        results = self.mModel.evaluate([self.mTestingData[0], self.mTestingData[1]], self.mTestingLabels)
        print(results)
        return

    def saveModel(self):
        if self.mModel is None:
            raise ValueError("Model has not been loaded, so there is nothing to save")
        self.mModel.save("Tensorflow.h5")
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
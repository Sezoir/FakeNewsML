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

        # self.mModel = ks.Sequential([
        #     lr.Embedding(VOCAB_SIZE, 32),
        #     lr.LSTM(32),
        #     lr.Dense(1, activation="sigmoid")
        # ])
        return

    def loadDatasets(self):
        self.mTrainingData = pd.read_csv(self.mTrainingPath, usecols=['title', 'text'], encoding='utf8').to_numpy(dtype=None)
        self.mTrainingLabels = pd.read_csv(self.mTrainingPath, usecols=['label'], encoding='utf8').to_numpy(dtype=None)
        self.createWordIndex()


        return

    def loadModel(self):
        if exists(Path("Tensorflow.h5")):
            self.mModel = ks.models.load_model("Tensorflow.h5")
            return

        self.mModel = ks.Sequential([
            lr.Embedding(65000, 32, input_length=200),#len(self.mToken.word_index)+1 #@todo: isnt giving the correct length
            lr.LSTM(32),
            lr.Dense(1, activation="sigmoid")
        ])

        self.mModel.summary()

        print(self.mModel.input_shape)

        self.mModel.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['acc'])

        return

    def fitModel(self):
        history = self.mModel.fit(self.mTrainingData, self.mTrainingLabels, epochs=10, validation_split=0.2)
        return

    def createWordIndex(self):
        token = prp.text.Tokenizer()
        token.fit_on_texts(self.mTrainingData[0] + self.mTrainingData[1]) # @todo: doesnt seem to fit all the words

        def textToIndex(text):
            tokens = [token.word_index[word] if word in token.word_index else 0 for word in prp.text.text_to_word_sequence(text)]
            return prp.sequence.pad_sequences([tokens], 200)[0].reshape(-1)

        x = []
        for arrayInd in range(len(self.mTrainingData)):
            self.mTrainingData[arrayInd][0] = textToIndex(self.mTrainingData[arrayInd][0])
            self.mTrainingData[arrayInd][1] = textToIndex(self.mTrainingData[arrayInd][1])
            x.append(self.mTrainingData[arrayInd][0]+self.mTrainingData[arrayInd][1])
        self.mTrainingData = x
        self.mTrainingData = np.asarray(self.mTrainingData, np.float32)
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
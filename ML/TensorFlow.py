from pathlib import Path
from os.path import exists
import numpy as np

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.utils as util
import tensorflow.keras.layers as lr
import tensorflow.keras.preprocessing as prp
import tensorflow_hub as hub

class TensorFlow:
    def __init__(self, trainingPath, validationPath, testingPath):
        # Initialise class variables
        self.mTrainingPath = trainingPath
        self.mValidationPath = validationPath
        self.mTestingPath = testingPath

        self.createWordIndex()

        # Load datasets
        self.loadDatasets()

        # self.mModel = ks.Sequential([
        #     lr.Embedding(VOCAB_SIZE, 32),
        #     lr.LSTM(32),
        #     lr.Dense(1, activation="sigmoid")
        # ])
        return

    def loadDatasets(self):
        BATCHSIZE = 32
        self.mTrainingData = tf.data.experimental.make_csv_dataset(str(self.mTrainingPath), batch_size=BATCHSIZE,
                                                            column_names=["title", "text", "subject", "date", "label"],
                                                            select_columns=["title", "text", "label"],
                                                            label_name="label")
        self.mValidationData = tf.data.experimental.make_csv_dataset(str(self.mValidationPath), batch_size=BATCHSIZE,
                                                            column_names=["title", "text", "subject", "date", "label"],
                                                            select_columns=["title", "text", "label"],
                                                            label_name="label")
        self.mTestingData = tf.data.experimental.make_csv_dataset(str(self.mTestingPath), batch_size=BATCHSIZE,
                                                            column_names=["title", "text", "subject", "date", "label"],
                                                            select_columns=["title", "text", "label"],
                                                            label_name="label")

        self.mTrainingData = self.mTrainingData.map(self.encode)
        self.mValidationData = self.mValidationData.map(self.encode)
        self.mTestingData = self.mTestingData.map(self.encode)

        for element in self.mTrainingData.take(1):
            print(element)
            print(len(element[0][0]))
            print(element[1][0])

        return

    def encode(self, data, label):
        for key in data:
            data[key] = self.mEmbed(data[key])
        # data = tf.concat(list(data.values()), -1)
        print(tf.shape(list(data)))
        return data, label

    def loadModel(self):
        if exists(Path("Tensorflow.h5")):
            self.mModel = ks.models.load_model("Tensorflow.h5")
            return

        self.mModel = ks.Sequential([ # lr.Embedding(input_dim=16000, output_dim=32), lr.Embedding(10000, input_length=500, output_dim=500),
            lr.LSTM(8, input_shape=(32,500)),
            lr.Dense(1, activation="sigmoid")
        ])

        self.mModel.summary()

        print(self.mModel.input_shape)

        self.mModel.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['acc'])

        return

    def fitModel(self):
        history = self.mModel.fit(self.mTrainingData, epochs=10, steps_per_epoch=10,
                                  validation_data=self.mValidationData, validation_steps=10)
        # history = self.mModel.fit(self.mTrainingData.batch(32), epochs=10, validation_data=self.mValidationData.batch(32))
        return

    def createWordIndex(self):
        self.mEmbed = hub.KerasLayer("https://tfhub.dev/google/Wiki-words-250-with-normalization/2",
                                input_shape=[], dtype=tf.string)

        # for x in trainingDataset.as_numpy_iterator():
        #     for key in x[0]:
        #         for ind in range(len(x[0][key])):
        #             print(x[0][key][ind])
        #             print(type(x[0][key][ind]))
        #             x[0][key][ind] = embed(x[0][key][ind])

        # for x in trainingDataset.take(1).as_numpy_iterator():
        #     for key in x[0]:
        #         print(embed(x[0][key]))
        #
        # for x in trainingDataset.take(1).as_numpy_iterator():
        #     for key in x[0]:
        #         print("Test: ", x[0][key])
        # print(tuple(test.take(1).as_numpy_iterator()))
        # print(type(test))
        return

    mTrainingPath = None
    mValidationPath = None
    mTestingPath = None
    mTrainingData = None
    mValidationData = None
    mTestingData = None
    mWordIndex = None
    mModel = None

    mEmbed = None
#!/usr/bin/env python3

from tensorflow import keras


def getEncoderOnly(model, inputLayerName, outputLayerName):
    inputLayer = model.get_layer(inputLayerName)
    outputLayer = model.get_layer(outputLayerName)

    encoderModel = keras.Model(inputs=inputLayer, outputs=outputLayer)
    return encoderModel

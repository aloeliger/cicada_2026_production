#!/usr/bin/env python3

from tensorflow import keras


def getEncoderOnly(model, inputLayerName, outputLayerName):
    inputLayer = model.input
    outputLayer = model.get_layer(outputLayerName).output

    encoderModel = keras.Model(inputs=inputLayer, outputs=outputLayer)
    return encoderModel


def getEncoderFromLayerList(model, layerList):
    listOfLayers = []

    for layerName in layerList:
        listOfLayers.append(model.get_layer(layerName))

    # encoderModel = keras.Model(inputs=inputLayer, outputs=x)
    encoderModel = keras.Sequential(listOfLayers)
    return encoderModel

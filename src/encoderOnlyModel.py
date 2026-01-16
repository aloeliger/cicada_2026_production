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


def getRebuiltEncoder(model, layerList):
    newSequentialLayers = keras.Sequential(name="EncoderOnly")
    newSequentialLayers.add(keras.layers.Input(shape=(18, 14, 1), name="new_input"))
    # originalModelLayers = []
    # for layerName in layerList:
    #     originalModelLayers.append(
    #         model.get_layer(layerName).output
    #     )

    # fullModelList = [
    #     keras.layers.Input(shape=(18,14,3)),
    # ] + originalModelLayers

    # encoderModel = keras.Sequential(fullModelList)
    for layerName in layerList:
        newSequentialLayers.add(model.get_layer(layerName))
    return newSequentialLayers

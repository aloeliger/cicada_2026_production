import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from sklearn.metrics import roc_curve

import trainTeacher


def loadInputsToDict(params, inputFn, background=False):
    if background:
        samples = dict(params["evaluation"]["samples"]["backgrounds"])
    else:
        samples = dict(params["evaluation"]["samples"]["signals"])

    inputs = {}
    for sampleName in samples:
        fileName = samples[sampleName]
        etGrid, tauBits, egBits = trainTeacher.loadFile(fileName)
        inputGrid = inputFn(etGrid, egBits, tauBits)
        inputs[sampleName] = inputGrid

    return inputs


def makePredictions(model, scoreFn, inputDict):
    outputDict = {}
    for sample in inputDict:
        inputs = inputDict[sample]
        samplePredictions = np.array(model.predict(inputs))
        scores = scoreFn(samplePredictions)
        outputDict[sample] = scores
    return outputDict


def getROCInfo(signalScores, backgroundScores):
    y_true = np.concatenate(
        [
            np.ones(len(signalScores)),
            np.zeros(len(backgroundScores)),
        ],
        axis=0,
    )
    y_pred = np.concatenate(
        [signalScores, backgroundScores],
        axis=0,
    )

    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    return fpr, tpr, threshold


def makeInformednessPlot(
    informednesses,
    rates,
    thresholds,
    signalName,
    outputName,
):
    hep.style.use("CMS")
    hep.cms.label("Preliminary", data=True, fontsize=11)

    plt.plot(rates, informednesses, label=f"{signalName}")

    plt.xlabel("Overall Trigger Rate [kHz]")
    plt.ylabel("Informedness")

    plt.xscale("log")
    plt.xlim(0.1, 10000)
    plt.ylim(-1.0, 1.0)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{outputName}.png", bbox_inches="tight")
    plt.savefig(f"{outputName}.pdf", bbox_inches="tight")

    plt.close()


def makeInformednessROC(
    rates, tpr, maxInformednessRate, maxInformednessTPR, signalSampleName, outputName
):
    hep.style.use("CMS")
    hep.cms.label("Preliminary", data=True, fontsize=11)

    plt.plot(rates, tpr, label=signalSampleName)
    plt.plot(
        [
            maxInformednessRate,
        ],
        [
            maxInformednessTPR,
        ],
        marker="o",
        linestyle="",
    )
    plt.xlabel("Overall Trigger Rate [kHz]")
    plt.ylabel("Fraction Signal Acceptance")

    plt.xscale("log")
    plt.xlim(0.1, 10000)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{outputName}.png", bbox_inches="tight")
    plt.savefig(f"{outputName}.pdf", bbox_inches="tight")

    plt.close()

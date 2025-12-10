#!/usr/bin/env python3
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from rich.console import Console
from rich.progress import track
from sklearn.metrics import confusion_matrix, roc_curve

from .utils import convert_eff_to_rate

console = Console()


def makeScores(model, caloRegions):
    predictions = model.predict(caloRegions)
    return np.array(predictions)


def makeROC(backgroundScores, signalScores):
    y_true = np.concatenate(
        [
            np.zeros((backgroundScores.shape[0],)),
            np.ones((signalScores.shape[0],)),
        ],
        axis=0,
    )
    y_score = np.concatenate([backgroundScores, signalScores], axis=0)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return fpr, tpr, thresholds


def makeROCPlot(fprs, tprs, thresholds, informednesses, outputHist):
    hep.style.use("CMS")
    hep.cms.text("Preliminary", loc=2)

    for signalSample in fprs:
        fpr = fprs[signalSample]
        tpr = tprs[signalSample]

        rates = convert_eff_to_rate(fpr)

        _ = plt.plot(rates, tpr, label=signalSample)

        # Plot point of maximum informedness
        informedness, i_fpr, i_tpr = informednesses[signalSample]
        i_rate = convert_eff_to_rate(i_fpr)
        _ = plt.plot(
            [
                i_rate,
            ],
            [
                i_tpr,
            ],
            marker="o",
        )
    plt.xlabel("Overall Trigger Rate [kHz]")
    plt.ylabel("Fraction Signal Acceptance")

    plt.xscale("log")
    plt.legend(loc="lower right")
    plt.savefig(outputHist)

    plt.close()


def getInformedness(backgroundScores, signalScores, threshold):
    y_true = np.concatenate(
        [
            np.zeros((backgroundScores.shape[0],)),
            np.ones((signalScores.shape[0],)),
        ],
        axis=0,
    )

    y_pred = np.concatenate(
        [
            np.where(backgroundScores > threshold, 1, 0),
            np.where(signalScores > threshold, 1, 0),
            # backgroundScores[backgroundScores>threshold].astype(np.int),
            # signalScores[signalScores>threshold].astype(np.int),
        ],
        axis=0,
    )

    # print(y_true.shape)
    # print(y_pred.shape)

    cfm = confusion_matrix(y_true, y_pred)

    tnr = cfm[0][0] / (cfm[0][0] + cfm[0][1])
    tpr = cfm[1][1] / (cfm[1][1] + cfm[1][0])

    informedness = tnr + tpr - 1.0
    return informedness


def getMaxInformedness(backgroundScores, signalScores, fprs, tprs, thresholds):
    bestIndex = 0
    maxThreshold = 0.0
    maxInformedness = -9999.0
    for index, threshold in track(
        enumerate(thresholds), description="Making informedness"
    ):
        informedness = getInformedness(backgroundScores, signalScores, threshold)
        if informedness > maxInformedness:
            bestIndex = index
            maxThreshold = threshold
            maxInformedness = informedness

    maxFpr = fprs[bestIndex]
    maxTpr = tprs[bestIndex]
    return maxInformedness, maxThreshold, maxFpr, maxTpr

#!/usr/bin/env python3
import numpy as np
import yaml
from rich.console import Console
from tensorflow import keras

from src import cicadaStudent_classic, evaluateStudent_classic, utils

console = Console()


def main(params):
    console.log("Making student evaluation")
    console.log("Getting list of files")
    fileList = utils.buildFileList(
        params["cicadaStudentCommon"]["fileDir"],
        params["cicadaStudentCommon"]["dataFileDilation"],
    )
    console.log("Loading data inputs")
    data_caloRegions, data_taubit, data_egbit, data_npvs, data_npvs_good = (
        cicadaStudent_classic.getInputs(fileList)
    )

    # Filter this down to the non-training set of data inputs
    _, _, data_caloRegions, _, _, _, _, _, _ = (
        cicadaStudent_classic.performDatasetSplitting(
            data_caloRegions, np.zeros(len(data_caloRegions))
        )
    )

    console.log("Loading Single W inputs")
    singleWFileList = utils.buildFileList(params["cicadaStudentCommon"]["singleWDir"])
    (
        singleW_caloRegions,
        singleW_taubit,
        singleW_egbit,
        singleW_npvs,
        singleW_npvs_good,
    ) = cicadaStudent_classic.getInputs(singleWFileList)

    classicModel = keras.models.load_model("data/cicadaStudent_classic/")

    # Make Scores
    dataScores = evaluateStudent_classic.makeScores(classicModel, data_caloRegions)
    singleW_scores = evaluateStudent_classic.makeScores(
        classicModel, singleW_caloRegions
    )

    # make ROCs
    singleW_fpr, singleW_tpr, singleW_thresholds = evaluateStudent_classic.makeROC(
        singleW_scores, dataScores
    )

    fprs = {"Single W": singleW_fpr}
    tprs = {"Single W": singleW_tpr}
    thresholds = {"Single W": singleW_thresholds}

    # Get informedness

    (
        singleW_bestInformedness,
        singleW_informedThreshold,
        singleW_informedFPR,
        singleW_informedTPR,
    ) = evaluateStudent_classic.getMaxInformedness(
        dataScores,
        singleW_scores,
        singleW_fpr,
        singleW_tpr,
        singleW_thresholds,
    )

    informednesses = {
        "Single W": (singleW_bestInformedness, singleW_informedFPR, singleW_informedTPR)
    }

    # Make the ROC Plot

    evaluateStudent_classic.makeROCPlot(
        fprs,
        tprs,
        thresholds,
        informednesses,
        outputHist=params["cicadaStudentClassic"]["ROCPlot"],
    )

    # print the informedness
    evaluateStudent_classic.printMaximumInformedness(informednesses)


if __name__ == "__main__":
    with open("params.yaml") as theFile:
        params = yaml.safe_load(theFile)
    main(params)

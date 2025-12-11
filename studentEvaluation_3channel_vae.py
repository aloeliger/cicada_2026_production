#!/usr/bin/env python3
import yaml
from rich.console import Console
from tensorflow import keras

from src import (
    cicadaStudent_classic,
    cicadaStudent_vae_3channel,
    evaluateStudent_classic,
)

console = Console()


def main(params):
    console.log("Making 3channel student evaluation")
    console.log("Loading data inputs")
    data_caloRegions, data_taubit, data_egbit, data_npvs, data_npvs_good = (
        cicadaStudent_classic.getInputs(params)
    )
    dataStudentInputs = cicadaStudent_vae_3channel.createStudentModelInputs(
        data_caloRegions, data_taubit, data_egbit
    )

    console.log("Loading Single W inputs")
    (
        singleW_caloRegions,
        singleW_taubit,
        singleW_egbit,
        singleW_npvs,
        singleW_npvs_good,
    ) = cicadaStudent_classic.getInputs(params, sampleList="single_W_list")
    singleWStudentInputs = cicadaStudent_vae_3channel.createStudentModelInputs(
        singleW_caloRegions,
        singleW_taubit,
        singleW_egbit,
    )

    vaeModel = keras.models.load_model("data/3channel_vae_student/")

    dataScores = evaluateStudent_classic.makeScores(vaeModel, dataStudentInputs)
    singleW_scores = evaluateStudent_classic.makeScores(vaeModel, singleWStudentInputs)

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
        outputHist=params["cicadaStudentVAE3Channel"]["ROCPlot"],
    )

    # print the informedness
    evaluateStudent_classic.printMaximumInformedness(informednesses)


if __name__ == "__main__":
    with open("params.yaml") as theFile:
        params = yaml.safe_load(theFile)
    main(params)

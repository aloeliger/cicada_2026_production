import argparse

import numpy as np
import yaml
from rich.console import Console
from tensorflow import keras

from src import evaluateModels as em
from src import utils as util

console = Console()


def main(params, args) -> None:
    console.log("Evaluating model")

    # get the models and their loss functions, and their translation to score functions
    def trivialScoreFunction(y_pred):
        return np.array(y_pred)

    def ThreeChannelInputFunction(caloRegions, egBits, tauBits):
        return np.concatenate([caloRegions, egBits, tauBits], axis=-1)

    def OneChannelInputFunction(caloRegions, egBits, tauBits):
        return caloRegions

    if args.model_style_group.oneChannelVAE:
        model = keras.models.load_model(
            "encoderOnlyModel_pythonUsability.keras",
        )

        def summed_square_scores(y_pred):
            scores = np.sum(np.array(y_pred) * np.array(y_pred), axis=1)
            return scores

        inputFn = OneChannelInputFunction
        scoreFn = summed_square_scores
    elif args.model_style_group.threeChannelClassicStudent:
        model = keras.models.load_model("cicadaStudent_classic_3channel.keras")
        inputFn = ThreeChannelInputFunction
        scoreFn = trivialScoreFunction
    else:
        model = keras.models.load_model("cicadaStudent_classic.keras")
        inputFn = OneChannelInputFunction
        scoreFn = trivialScoreFunction

    # Get all the inputs and translate the inputs into model inputs if necessary
    backgroundModelInputs = em.loadInputsToDict(params, inputFn, background=True)
    signalModelInputs = em.loadInputsToDict(params, inputFn, background=False)

    # make model predictions per sample
    backgroundPredictions = em.makePredictions(model, scoreFn, backgroundModelInputs)
    signalPredictions = em.makePredictions(model, scoreFn, signalModelInputs)

    # Make ROCS + Informedness
    for backgroundSampleName in backgroundPredictions:
        for signalSampleName in signalPredictions:
            console.log(f"Processing ROC for {backgroundSampleName}/{signalSampleName}")
            fpr, tpr, threshold = em.getROCInfo(
                signalScores=signalPredictions[signalSampleName],
                backgroundScores=backgroundPredictions[backgroundSampleName],
            )

            rates = util.convert_eff_to_rate(fpr)

            informedness = tpr - fpr
            maxInformednessIndex = np.argmax(informedness)
            # maxInformedness = informedness[maxInformednessIndex]
            # maxThreshold = threshold[maxInformednessIndex]
            maxFpr = fpr[maxInformednessIndex]
            maxTpr = tpr[maxInformednessIndex]
            maxRate = util.convert_eff_to_rate(maxFpr)

            em.makeInformednessPlot(
                informedness,
                rates,
                threshold,
                signalSampleName,
                outputName=f"data/{backgroundSampleName}_{signalSampleName}_informedness",
            )

            em.makeInformednessROC(
                rates,
                tpr,
                maxRate,
                maxTpr,
                signalSampleName,
                outputName=f"data/{backgroundSampleName}_{signalSampleName}_informednessROC",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models via ROC cruves")

    model_style_group = parser.add_mutually_exclusive_group(required=True)

    model_style_group.add_argument("--classicStudent", action="store_true")
    model_style_group.add_argument("--threeChannelClassicStudent", action="store_true")
    model_style_group.add_argument("--oneChannelVAE", action="store_true")

    with open("params.yaml") as theFile:
        params = yaml.safe_load(theFile)

    args = parser.parse_args()

    main(params, args)

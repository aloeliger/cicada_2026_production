#!/usr/bin/env python3
"""Make a script to process down ROOT files into things useful for machine learning"""

import yaml
from rich.console import Console
from sklearn.model_selection import train_test_split

import src.trainingFiles as trainingFiles

console = Console()


def main(params):
    for fileType in params["files"]:
        console.log(f"Processing file type: {fileType}")
        goodRuns = None
        if fileType == "dataFiles":
            goodRuns = params["goodRuns"]
        etGrids, tauGrids, egGrids = trainingFiles.processFiles(
            fileType=fileType,
            filePath=params["files"][fileType],
            limitInputs=params["limitInputs"],
            goodRuns=goodRuns,
        )

        if fileType == "dataFiles":
            (
                etGrids_train,
                etGrids_other,
                tauGrids_train,
                tauGrids_other,
                egGrids_train,
                egGrids_other,
            ) = train_test_split(
                etGrids, tauGrids, egGrids, random_state=123, test_size=0.666
            )
            (
                etGrids_student,
                etGrids_test,
                tauGrids_student,
                tauGrids_test,
                egGrids_student,
                egGrids_test,
            ) = train_test_split(
                etGrids_other,
                tauGrids_other,
                egGrids_other,
                random_state=123,
                test_size=0.333 / 0.666,
            )

            trainingFiles.saveGrids(
                etGrids_train,
                tauGrids_train,
                egGrids_train,
                outputPath=f"data/{fileType}_training_output.h5",
            )

            trainingFiles.saveGrids(
                etGrids_student,
                tauGrids_student,
                egGrids_student,
                outputPath=f"data/{fileType}_studentTraining_output.h5",
            )

            trainingFiles.saveGrids(
                etGrids_test,
                tauGrids_test,
                egGrids_test,
                outputPath=f"data/{fileType}_output.h5",
            )

        else:
            trainingFiles.saveGrids(
                etGrids,
                tauGrids,
                egGrids,
                outputPath=f"data/{fileType}_output.h5",
            )


if __name__ == "__main__":
    with open("params.yaml") as theFile:
        params = yaml.safe_load(theFile)
    main(params)

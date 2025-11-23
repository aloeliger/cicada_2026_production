#!/usr/bin/env python3
"""Make a script to process down ROOT files into things useful for machine learning"""

import yaml
from rich.console import Console

import cicada_2026_production.src.trainingFiles as trainingFiles

console = Console()


def main(params):
    for fileType in params["files"]:
        console.log(f"Processing file type: {fileType}")
        etGrids, tauGrids, egGrids = trainingFiles.processFiles(
            fileType=fileType,
            filePath=params["files"][fileType],
        )

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

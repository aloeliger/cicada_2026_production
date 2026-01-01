#!/usr/bin/env python3
"""Perform training of the CICADA teacher model"""

import yaml
from rich.console import Console
from sklearn.model_selection import train_test_split

from src import trainTeacher

console = Console()


def main(params):
    dataPath = params["inputFiles"]["dataFile"]
    caloRegions, tauBits, egBits = trainTeacher.loadFile(dataPath)

    teacherModel = trainTeacher.makeTeacherModel(
        inputShape=caloRegions.shape[1:],
        alpha=params["alpha"],
        beta=params["beta"],
    )

    train_grids, test_grids = train_test_split(
        caloRegions,
        test_size=0.2,
        random_state=123,
    )

    trainTeacher.trainModel(teacherModel, inputs=train_grids, test_inputs=test_grids)


if __name__ == "__main__":
    with open("params.yaml") as theFile:
        params = yaml.safe_load(theFile)

    main(params)

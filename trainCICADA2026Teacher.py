#!/usr/bin/env python3
"""Perform training of the CICADA teacher model"""

import yaml
from rich.console import Console
from sklearn.model_selection import train_test_split

import cicada_2026_production.src.trainTeacher as trainTeacher

console = Console()


def main(params):
    dataPath = params["files"]["dataFiles"]
    dataGrids = trainTeacher.loadFile(dataPath)

    teacherModel = trainTeacher.makeTeacherModel(
        inputShape=dataGrids.shape[1:],
        alpha=params["alpha"],
        beta=params["beta"],
    )

    train_grids, test_grids = train_test_split(
        dataGrids,
        test_size=0.2,
        random_state=123,
    )

    trainTeacher.trainModel(teacherModel, inputs=train_grids, test_inputs=test_grids)


if __name__ == "__main__":
    with open("params.yaml") as theFile:
        params = yaml.safe_load(theFile)

    main(params)

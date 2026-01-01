#!/usr/bin/env python3
"""Perform training of the CICADA teacher model"""

import argparse

import numpy as np
import yaml
from rich.console import Console
from sklearn.model_selection import train_test_split

from src import trainTeacher

console = Console()


def main(params, args):
    dataPath = params["inputFiles"]["dataFile"]
    caloRegions, tauBits, egBits = trainTeacher.loadFile(dataPath)

    dataGrids = caloRegions
    if args.use3Channels:
        dataGrids = np.concatenate([caloRegions, tauBits, egBits], axis=-1)

    teacherModel = trainTeacher.makeTeacherModel(
        inputShape=dataGrids.shape[1:],
        alpha=params["alpha"],
        beta=params["beta"],
        use3Channels=args.use3Channels,
    )

    train_grids, test_grids = train_test_split(
        dataGrids,
        test_size=0.2,
        random_state=123,
    )

    trainTeacher.trainModel(
        teacherModel,
        inputs=train_grids,
        test_inputs=test_grids,
        use3Channels=args.use3Channels,
    )


if __name__ == "__main__":
    with open("params.yaml") as theFile:
        params = yaml.safe_load(theFile)

    parser = argparse.ArgumentParser(description="Make teacher models")
    parser.add_argument("--use3Channels", action="store_true")

    args = parser.parse_args()

    main(params, args)

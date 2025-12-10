#!/usr/bin/env python3
import os
import tempfile

import numpy as np
import pytest
from sklearn.metrics import roc_curve

import cicada_2026_production.src.evaluateStudent_classic as esc


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        ((np.zeros((1,)), np.ones((1,))), [0.0, 1.0]),
        ((np.ones((1,)), np.ones((1,))), [1.0, 1.0]),
        ((np.ones((1,)), np.zeros((1,))), [1.0, 0.0]),
    ],
)
def test_makeROC(input_value, expected_output):
    backgroundScores, signalScores = input_value

    fpr, tpr, thresholds = esc.makeROC(backgroundScores, signalScores)
    index = np.where(thresholds == 1.0)
    print(fpr)
    print(tpr)
    print(thresholds)
    assert fpr[index] == expected_output[0]
    assert tpr[index] == expected_output[1]


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        ((np.zeros((1,)), np.ones((1,))), 1.0),
        (
            (np.ones((1,)), np.ones((1,))),
            0.0,
        ),
        ((np.ones((1,)), np.zeros((1,))), -1.0),
    ],
)
def test_getInformedness(input_value, expected_output):
    backgroundScores, signalScores = input_value
    threshold = 0.5

    informedness = esc.getInformedness(backgroundScores, signalScores, threshold)
    assert informedness == expected_output


def testmakeROCPlot():
    fprs = {"fake": np.array([0.0, 0.5])}
    tprs = {"fake": np.array([0.0, 1.0])}
    thresholds = {"fake": np.array([0.0, 0.5])}
    informednesses = {"fake": (1.0, 0.0, 1.0)}

    with tempfile.TemporaryDirectory() as theDir:
        esc.makeROCPlot(fprs, tprs, thresholds, informednesses, theDir + "/fake.png")

        assert os.listdir(theDir)


# TODO: this test really should be better
# actually calculate the (max) informedness of this.
def testGetMaxInformedness():
    backgroundLabels = np.zeros((5,))
    signalLabels = np.ones((5,))
    backgroundScores = (np.arange(5)) / 7
    signalScores = (np.arange(5) + 2) / 7

    fprs, tprs, thresholds = roc_curve(
        np.concatenate([backgroundLabels, signalLabels], axis=0),
        np.concatenate([backgroundScores, signalScores], axis=0),
    )

    _, _, _, _ = esc.getMaxInformedness(
        backgroundScores, signalScores, fprs, tprs, thresholds
    )

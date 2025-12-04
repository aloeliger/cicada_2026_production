#!/usr/bin/env python3

import numpy as np
import pytest
from tensorflow.keras.losses import MeanSquaredError

import cicada_2026_production.src.cicadaStudent_vae_3channel as cicada_3channel


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        ((np.ones((18, 14, 1)), np.ones((18, 14, 1)), np.ones((18, 14, 1))), 7),
        ((np.ones((18, 14, 1)), np.ones((18, 14, 1)), np.zeros((18, 14, 1))), 6),
        ((np.ones((18, 14, 1)), np.zeros((18, 14, 1)), np.zeros((18, 14, 1))), 4),
        ((np.zeros((18, 14, 1)), np.ones((18, 14, 1)), np.ones((18, 14, 1))), 3),
        ((np.zeros((18, 14, 1)), np.ones((18, 14, 1)), np.zeros((18, 14, 1))), 2),
        ((np.zeros((18, 14, 1)), np.zeros((18, 14, 1)), np.ones((18, 14, 1))), 1),
        ((np.zeros((18, 14, 1)), np.zeros((18, 14, 1)), np.zeros((18, 14, 1))), 0),
    ],
)
def testCreateStudentModelInputs(input_value, expected_output):
    caloRegions, tauBits, egBits = input_value

    outputs = cicada_3channel.createStudentModelInputs(caloRegions, tauBits, egBits)
    assert outputs[0][0][0] == expected_output


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        ((np.ones((1, 2, 2, 1)), np.ones((1, 2, 2, 1))), 0.0),
        ((np.zeros((1, 2, 2, 1)), np.ones((1, 2, 2, 1))), 1.0),
    ],
)
def test_soft_dice_loss(input_value, expected_output, mocker):
    y_pred, y_true = input_value
    loss = cicada_3channel.soft_dice_loss(y_true, y_pred, smooth=0.0)

    assert loss == expected_output


def test_hybrid_bce_dice_loss():
    ones = np.ones((1, 2, 2, 1))
    zeros = np.zeros((1, 2, 2, 1))

    perfect_loss = np.array(cicada_3channel.hybrid_bce_dice_loss(ones, ones))
    worst_loss = np.array(cicada_3channel.hybrid_bce_dice_loss(ones, zeros))

    print(perfect_loss)
    print(worst_loss)

    assert perfect_loss < worst_loss


def test_makeTargets(mocker):
    mockTeacher = mocker.Mock()
    mockTeacher.predict.return_value = [
        np.zeros((10, 18, 14, 1)),
        np.zeros((10, 18, 14, 1)),
        np.zeros((10, 18, 14, 1)),
    ]
    # mockTeacher.loss = cicada_3channel.hybrid_bce_dice_loss
    mockTeacher.loss = {
        "output_energy": MeanSquaredError(reduction="none"),
        "output_tau": cicada_3channel.hybrid_bce_dice_loss,
        "output_egamma": cicada_3channel.hybrid_bce_dice_loss,
    }

    ones = np.ones((10, 18, 14, 1))
    _ = cicada_3channel.makeTargets(mockTeacher, ones, ones, ones)

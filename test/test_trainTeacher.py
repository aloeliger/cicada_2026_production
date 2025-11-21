from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
from tensorflow import keras

import cicada_2026_production.src.trainTeacher as trainTeacher


@pytest.fixture
def etGrids():
    return np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]])


@pytest.fixture
def tauBitGrids():
    return np.array(
        [
            [
                [
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                ],
            ]
        ]
    )


@pytest.fixture
def egBitGrids():
    return np.array(
        [
            [
                [
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                ],
                [1, 1, 1],
            ]
        ]
    )


# TODO, this test pases by default, and only fails on code errors
# this should be adjusted to be a real test
def test_loadFile(mocker):
    mock_file = MagicMock()

    mocker.patch("h5py.open", return_value=mock_file)

    _ = trainTeacher.loadFile("dummyPath/")


def test_make_mse_bce_loss(etGrids, tauBitGrids, egBitGrids, mocker):
    lossFn = trainTeacher.make_mse_bce_loss(1.0, 1.0)
    assert isinstance(lossFn, Callable)

    y_true = np.concatenate(
        [
            etGrids.reshape((-1, 3, 3, 1)).astype(np.float64),
            tauBitGrids.reshape((-1, 3, 3, 1)).astype(np.float64),
            egBitGrids.reshape((-1, 3, 3, 1)).astype(np.float64),
        ],
        axis=-1,
    )
    y_pred = y_true
    _ = lossFn(y_true, y_pred)


def test_makeModel(mocker):
    mocker.patch(
        "cicada_2026_production.src.trainTeacher.make_mse_bce_loss",
        lambda alpha, beta: None,
    )

    model = trainTeacher.makeTeacherModel(
        inputShape=(3, 3, 1),
        alpha=1.0,
        beta=1.0,
    )

    assert isinstance(model, keras.Model)

import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest

import cicada_2026_production.src.evaluateModels as evaluateModels


@pytest.fixture
def mockParams():
    return {"evaluation": {"samples": {"backgrounds": {"dummy": "dummyFile.h5"}}}}


def test_loadInputsToDict(mocker, mockParams):
    mock_input_tuple = (
        np.ones((18, 14, 1)),
        np.ones((18, 14, 1)),
        np.ones((18, 14, 1)),
    )

    mocker.patch("trainTeacher.loadFile", return_value=mock_input_tuple)

    def ThreeChannelInputFunction(caloRegions, egBits, tauBits):
        return np.concatenate([caloRegions, egBits, tauBits], axis=-1)

    outputs = evaluateModels.loadInputsToDict(
        params=mockParams, inputFn=ThreeChannelInputFunction, background=True
    )

    assert isinstance(outputs, dict)
    assert "dummy" in outputs
    assert outputs["dummy"].shape == (18, 14, 3)


def test_makePredictions(mocker):
    model = MagicMock()
    model_predict = MagicMock()
    model_predict.return_value = np.ones((10, 1))
    model.predict = model_predict

    mockInputs = {"dummy": np.ones((10, 18, 14, 1))}

    def trivialScoreFunction(y_pred):
        return np.array(y_pred)

    outputDict = evaluateModels.makePredictions(
        model, scoreFn=trivialScoreFunction, inputDict=mockInputs
    )

    assert isinstance(outputDict, dict)
    assert "dummy" in outputDict


def test_getROCInfo():
    mockSignalScores = np.ones(5)
    mockBackgroundScores = np.zeros(5)
    info = evaluateModels.getROCInfo(
        signalScores=mockSignalScores,
        backgroundScores=mockBackgroundScores,
    )

    assert isinstance(info, tuple)

    tpr = info[1]
    fpr = info[0]

    assert np.mean(fpr) < np.mean(tpr)


def test_makeInformednessPlot():
    informednesses = np.arange(10) / 10.0
    mockRates = np.arange(10)
    mockThreshold = np.arange(10)
    signalName = "dummy"

    with tempfile.TemporaryDirectory() as tempdir:
        outputName = f"{tempdir}/dummy"
        evaluateModels.makeInformednessPlot(
            informednesses,
            mockRates,
            mockThreshold,
            signalName,
            outputName,
        )

        assert os.listdir(tempdir)


def test_makeInformednessROC():
    mockRates = np.arange(5) * 1.0
    mockTpr = np.arange(5) / 5.0
    mockMIRate = 1.0
    mockMITPR = 0.99
    signalSampleName = "dummy"

    with tempfile.TemporaryDirectory() as tempdir:
        outputName = f"{tempdir}/dummy"
        evaluateModels.makeInformednessROC(
            mockRates, mockTpr, mockMIRate, mockMITPR, signalSampleName, outputName
        )

        assert os.listdir(tempdir)

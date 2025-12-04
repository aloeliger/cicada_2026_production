#!/usr/bin/env python3

import os
import tempfile
from unittest.mock import MagicMock

import numpy as np

import cicada_2026_production.src.cicadaStudent_classic as cicadaStudent_classic


def test_getInputs(mocker):
    testParams = {"cicadaStudentCommon": {"fileList": ["fileOne.h5"]}}

    mockData = {
        "CaloRegions": {
            "et": [1, 2, 3],
            "taubit": [0, 1, 0],
            "egbit": [1, 0, 0],
        },
        "PV_npvs": [10, 20, 30],
        "PV_npvsGood": [4, 5, 6],
    }

    fileEnterReturn = MagicMock()
    fileEnterReturn.__getitem__.side_effect = mockData.__getitem__
    fileEnterReturn.__contains__.side_effect = mockData.__contains__

    fileMock = MagicMock()
    fileMock.__enter__.return_value = fileEnterReturn

    mocker.patch("h5py.File", return_value=fileMock)
    caloRegions, taubit, egbit, npvs, npvs_good = cicadaStudent_classic.getInputs(
        testParams
    )

    assert np.array_equal(caloRegions, np.array(mockData["CaloRegions"]["et"]))
    assert np.array_equal(taubit, np.array(mockData["CaloRegions"]["taubit"]))
    assert np.array_equal(egbit, np.array(mockData["CaloRegions"]["egbit"]))
    assert np.array_equal(npvs, np.array(mockData["PV_npvs"]))
    assert np.array_equal(npvs_good, np.array(mockData["PV_npvsGood"]))


def testGetModel():
    _ = cicadaStudent_classic.getModel(inputShape=(252,))


def testMakeTargets(mocker):
    mockTeacher = mocker.Mock()
    caloRegions = np.ones((10, 18, 14)) * 1.0
    tauBit = np.zeros((10, 18, 14)) * 0.0
    egBit = np.ones((10, 18, 14)) * 1.0

    mockTeacher.predict.return_value = 0.5 * caloRegions.reshape((-1, 18, 14, 1))

    targets = cicadaStudent_classic.makeTargets(
        mockTeacher,
        caloRegions,
        tauBit,
        egBit,
    )

    targets = np.array(targets)
    assert targets.shape[0] == 10
    # assert np.all(targets != 0)
    assert targets[0] == np.clip(32.0 * np.log((0.5) ** 2), a_min=0.0, a_max=256.0)


def testMakeScoreWeights(mocker):
    targets = np.arange(100)
    targets[3] = targets[4]

    with tempfile.TemporaryDirectory() as tempdir:
        targetWeights = cicadaStudent_classic.makeScoreWeights(
            targets, tempdir + "/tempfile.h5"
        )
        assert os.listdir(tempdir)
    assert targetWeights[0] > targetWeights[3]


def testTrainStudentModel(mocker):
    mockModel = mocker.Mock()

    caloRegions = np.tile(np.array([[1, 2, 3]]), (1000, 1))
    targets = np.tile(np.array([[1.0, 1.0, 1.0]]), (1000, 1))

    cicadaStudent_classic.trainStudentModel(
        mockModel,
        caloRegions,
        targets,
    )

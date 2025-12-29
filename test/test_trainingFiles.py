import os
import tempfile

import numpy as np
import pytest
import ROOT

import cicada_2026_production.src.trainingFiles as trainingFiles


@pytest.fixture
def etInput():
    return np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]])


@pytest.fixture
def iphi():
    return np.array([[0, 2, 1, 0, 1, 2, 2, 1, 0]])


@pytest.fixture
def ieta():
    return np.array([[0, 0, 0, 1, 1, 1, 2, 2, 2]])


@pytest.fixture
def tauBits():
    return np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0]])


@pytest.fixture
def egBits():
    return np.array([[0, 0, 0, 0, 0, 0, 1, 1, 1]])


@pytest.fixture
def mock_batch(mocker, etInput, iphi, ieta, tauBits, egBits):
    mock_batch = mocker.Mock()
    mock_batch.Regions_taubit = tauBits
    mock_batch.Regions_egbit = egBits
    mock_batch.Regions_et = etInput
    mock_batch.Regions_iphi = iphi
    mock_batch.Regions_ieta = ieta
    return mock_batch


def test_reshapeInput(etInput, iphi, ieta):
    reshapedInput = trainingFiles.reshapeInput(
        inputs=etInput,
        iphi=iphi,
        ieta=ieta,
        total_iphi=3,
        total_ieta=3,
    )

    assert reshapedInput[0, 0, 0] == 0
    assert reshapedInput[0, 2, 0] == 1
    assert reshapedInput[0, 1, 0] == 2
    assert reshapedInput[0, 0, 1] == 3
    assert reshapedInput[0, 1, 1] == 4
    assert reshapedInput[0, 2, 1] == 5
    assert reshapedInput[0, 2, 2] == 6
    assert reshapedInput[0, 1, 2] == 7
    assert reshapedInput[0, 0, 2] == 8


def test_reshapeInput_2D():
    test_regions = np.arange(252).reshape(1, -1)
    test_phi = test_regions // 14
    test_ieta = test_regions % 14

    test_regions = np.tile(test_regions, (100, 1))
    test_phi = np.tile(test_phi, (100, 1))
    test_ieta = np.tile(test_ieta, (100, 1))

    expected_output = test_regions.reshape((-1, 18, 14))

    print(test_regions[0].reshape((18, 14)))
    print(test_regions[0].shape)
    print(test_phi[0].reshape((18, 14)))
    print(test_ieta[0].reshape((18, 14)))

    nprng = np.random.default_rng(42)

    stacked_input = np.stack([test_regions, test_phi, test_ieta], axis=2)
    nprng.shuffle(stacked_input, axis=1)
    test_regions = stacked_input[:, :, 0]
    test_phi = stacked_input[:, :, 1]
    test_ieta = stacked_input[:, :, 2]

    # nprng.shuffle(test_regions, axis=1)
    # nprng.shuffle(test_phi, axis=1)
    # nprng.shuffle(test_ieta, axis=1)

    reshaped_input = trainingFiles.reshapeInput(
        test_regions,
        test_ieta,
        test_phi,
    )

    print(test_regions[0].reshape((18, 14)))
    print(test_regions[0].shape)
    print(test_phi[0].reshape((18, 14)))
    print(test_ieta[0].reshape((18, 14)))

    print(reshaped_input[0])
    print(reshaped_input[0].shape)
    print(expected_output[0])
    print(expected_output[0].shape)

    assert np.array_equal(reshaped_input, expected_output)


def test_processBatch(mock_batch):
    reshapedEts, reshapedTauBits, reshapedEGBits = trainingFiles.processBatch(
        mock_batch
    )

    # these work, and put things in the right place,
    # but the returned grids are too large because
    # it uses default sizing in the function
    assert reshapedEts[0, 0, 0] == 0
    assert reshapedEts[0, 2, 0] == 1
    assert reshapedEts[0, 1, 0] == 2
    assert reshapedEts[0, 0, 1] == 3
    assert reshapedEts[0, 1, 1] == 4
    assert reshapedEts[0, 2, 1] == 5
    assert reshapedEts[0, 2, 2] == 6
    assert reshapedEts[0, 1, 2] == 7
    assert reshapedEts[0, 0, 2] == 8

    assert reshapedTauBits[0, 0, 0] == 0
    assert reshapedTauBits[0, 2, 0] == 0
    assert reshapedTauBits[0, 1, 0] == 0
    assert reshapedTauBits[0, 0, 1] == 1
    assert reshapedTauBits[0, 1, 1] == 1
    assert reshapedTauBits[0, 2, 1] == 1
    assert reshapedTauBits[0, 2, 2] == 0
    assert reshapedTauBits[0, 1, 2] == 0
    assert reshapedTauBits[0, 0, 2] == 0

    assert reshapedEGBits[0, 0, 0] == 0
    assert reshapedEGBits[0, 2, 0] == 0
    assert reshapedEGBits[0, 1, 0] == 0
    assert reshapedEGBits[0, 0, 1] == 0
    assert reshapedEGBits[0, 1, 1] == 0
    assert reshapedEGBits[0, 2, 1] == 0
    assert reshapedEGBits[0, 2, 2] == 1
    assert reshapedEGBits[0, 1, 2] == 1
    assert reshapedEGBits[0, 0, 2] == 1


def test_processFiles(mock_batch, mocker):
    mocker.patch(
        "uproot.iterate",
        return_value=[
            mock_batch,
        ],
    )
    mocker.patch("utils.buildFileList", return_value=["/tmp/NO_FILE.root"])
    mocker.patch.object(ROOT.TChain, "Add", return_value=None)
    mocker.patch.object(ROOT.TChain, "GetEntries", return_value=100)

    etGrids, tauBitGrids, egBitGrids = trainingFiles.processFiles(
        "dummy",
        "/tmp/",
    )

    assert etGrids.shape == (1, 18, 14)
    assert tauBitGrids.shape == (1, 18, 14)
    assert egBitGrids.shape == (1, 18, 14)


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


def test_saveGrids(etGrids, tauBitGrids, egBitGrids):
    with tempfile.TemporaryDirectory() as tempdir:
        trainingFiles.saveGrids(
            etGrids, tauBitGrids, egBitGrids, outputPath=f"{tempdir}/temp_file.h5"
        )

        assert os.listdir(tempdir)


def test_makeGoodRunCut():
    goodRuns = [1, 2, 3]
    goodRunCut = trainingFiles.makeGoodRunCut(goodRuns)
    assert goodRunCut == "run == 1 || run == 2 || run == 3"

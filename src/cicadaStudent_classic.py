#!/usr/bin/env python3
import h5py
import numpy as np
import qkeras
from rich.console import Console
from rich.progress import track
from sklearn.model_selection import train_test_split
from tensorflow import keras

console = Console()


def getInputs(params, sampleList="fileList"):
    listOfFiles = params["cicadaStudentCommon"][sampleList]
    with h5py.File(listOfFiles[0]) as theFile:
        caloRegions = np.array(theFile["CaloRegions"]["et"])
        taubit = np.array(theFile["CaloRegions"]["taubit"])
        egbit = np.array(theFile["CaloRegions"]["egbit"])
        npvs = np.array(theFile["PV_npvs"])
        npvs_good = np.array(theFile["PV_npvsGood"])

    for fileName in track(listOfFiles[1:], console=console):
        # console.log(fileName)
        try:
            with h5py.File(fileName, "r") as theFile:
                caloRegions = np.concatenate(
                    [
                        caloRegions,
                        np.array(theFile["CaloRegions"]["et"]),
                    ],
                    axis=0,
                )
                taubit = np.concatenate(
                    [
                        taubit,
                        np.array(theFile["CaloRegions"]["taubit"]),
                    ],
                    axis=0,
                )
                egbit = np.concatenate(
                    [egbit, np.array(theFile["CaloRegions"]["egbit"])], axis=0
                )
                npvs = np.concatenate(
                    [
                        npvs,
                        np.array(theFile["PV_npvs"]),
                    ],
                    axis=0,
                )
                npvs_good = np.concatenate(
                    [npvs_good, np.array(theFile["PV_npvsGood"])], axis=0
                )
        except OSError:
            console.log(f"[red]Warning, failed to process file: {fileName}[/red]")
    caloRegions = caloRegions.reshape((-1, 252))
    taubit = taubit.reshape((-1, 252))
    egbit = egbit.reshape((-1, 252))
    return caloRegions, taubit, egbit, npvs, npvs_good


def getTeacherModel(fileLocation):
    return keras.models.load_model(fileLocation)


def getModel(inputShape):
    inputs = keras.layers.Input(shape=inputShape, name="student_input")
    reshape = keras.layers.Reshape((18, 14, 1), name="reshape")(inputs)
    conv_1 = qkeras.QConv2D(
        4,
        (2, 2),
        strides=2,
        padding="valid",
        use_bias=False,
        kernel_quantizer=qkeras.quantized_bits(12, 3, 1, alpha=1.0),
        name="conv",
    )(reshape)
    act_1 = qkeras.QActivation("quantized_relu(10, 6)", name="relu0")(conv_1)
    flat = keras.layers.Flatten(name="flatten")(act_1)
    drop_1 = keras.layers.Dropout(1 / 9)(flat)
    denseBN_1 = qkeras.QDenseBatchnorm(
        16,
        kernel_quantizer=qkeras.quantized_bits(8, 1, 1, alpha=1.0),
        bias_quantizer=qkeras.quantized_bits(8, 3, 1, alpha=1.0),
        name="dense1",
    )(drop_1)
    act_2 = qkeras.QActivation("quantized_relu(10, 6)", name="relu1")(denseBN_1)
    drop_2 = keras.layers.Dropout(1 / 8)(act_2)
    dense_1 = qkeras.QDense(
        1,
        kernel_quantizer=qkeras.quantized_bits(12, 3, 1, alpha=1.0),
        use_bias=False,
        name="dense2",
    )(drop_2)
    outputs = qkeras.QActivation("quantized_relu(16, 8)", name="outputs")(dense_1)

    model = keras.Model(inputs, outputs, name="cicada-v2")
    model.compile(
        optimizer="nadam",
        loss="mse",
        weighted_metrics=[
            "mae",
        ],
    )
    return model


def makeTargets(teacher, caloRegions, taubit, egbit):
    # print(caloRegions)
    teacherPredictions = np.array(teacher.predict(caloRegions.reshape((-1, 18, 14, 1))))
    # print(teacherPredictions)

    loss = np.mean(
        (
            np.array(caloRegions).reshape((-1, 18, 14, 1))
            - np.array(teacherPredictions).reshape((-1, 18, 14, 1))
        )
        ** 2,
        axis=(1, 2, 3),
    )

    # print(loss)
    loss = np.clip(32.0 * np.log(loss), a_min=0.0, a_max=256.0)
    # print(loss)
    # print(loss.shape)

    return loss


def makeScoreWeights(targets, outputFile):
    scoreHistogram, binEdges = np.histogram(
        targets,
        bins=100,
        density=True,
        range=(np.min(targets) - 0.1, np.max(targets) + 0.1),
    )
    # scoreHistogram = scoreHistogram + 1e-12

    # choose bin edges and bins such that we always have _something_
    # in that bin
    binEdges = binEdges[:-1][scoreHistogram > 0]
    scoreHistogram = scoreHistogram[scoreHistogram > 0]

    reweightHistogram = 1.0 / scoreHistogram
    reweightHistogram = reweightHistogram / np.mean(reweightHistogram)
    reweightHistogram = np.clip(reweightHistogram, a_min=None, a_max=20.0)

    targetWeightBins = np.digitize(targets, binEdges)
    targetWeightBins = targetWeightBins - 1
    targetWeights = reweightHistogram[targetWeightBins]

    console.log("Score histogram")
    console.log(scoreHistogram)
    console.log("Weighting histogram")
    console.log(reweightHistogram)
    console.log("Bin edges")
    console.log(binEdges)

    with h5py.File(outputFile, "w") as theFile:
        theFile.create_dataset("scoreHistogram", data=scoreHistogram)
        theFile.create_dataset("weightHistogram", data=reweightHistogram)
        theFile.create_dataset("binEdges", data=binEdges)

    return targetWeights


def trainStudentModel(model, caloRegions, targets, weights=None):
    if weights is None:
        train_caloRegions, testval_caloRegions, train_targets, testval_targets = (
            train_test_split(caloRegions, targets, test_size=0.2, random_state=42)
        )

        test_caloRegions, val_caloRegions, test_targets, val_targets = train_test_split(
            testval_caloRegions, testval_targets, test_size=0.1 / 0.2, random_state=123
        )
        train_weights, val_weights, test_weights = (
            None,
            None,
            None,
        )

    else:
        (
            train_caloRegions,
            testval_caloRegions,
            train_targets,
            testval_targets,
            train_weights,
            testval_weights,
        ) = train_test_split(
            caloRegions, targets, weights, test_size=0.2, random_state=42
        )

        (
            test_caloRegions,
            val_caloRegions,
            test_targets,
            val_targets,
            test_weights,
            val_weights,
        ) = train_test_split(
            testval_caloRegions,
            testval_targets,
            testval_weights,
            test_size=0.1 / 0.2,
            random_state=123,
        )

    keras.utils.set_random_seed(123)
    model.fit(
        train_caloRegions,
        train_targets,
        epochs=1000,
        validation_data=(val_caloRegions, val_targets, val_weights),
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                "data/cicadaStudent_classic", save_best_only=True
            ),
            keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=10),
            keras.callbacks.CSVLogger("data/logs/classic_student_log.csv"),
        ],
        sample_weight=train_weights,
        batch_size=128,
    )

    console.log("Evaluation")
    model.evaluate(test_caloRegions, test_targets, sample_weight=test_weights)

#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from rich.console import Console
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.losses import BinaryCrossentropy

console = Console()

# Okay. 10 bits are used for the integer ET
# 1 bit for the tau bit
# and 1 bit for the EG bit
# So the way I am going to pack this model is like so:
# Bit:  1  2  3  4  5  6  7  8  9  10  11  12
#       |     Calo region bits     |   tau eg


def createStudentModelInputs(caloRegions, tauBits, egBits):
    caloInt = caloRegions.astype(np.int64)
    caloInt = caloInt << 2

    tauInt = tauBits.astype(np.int64)
    tauInt = tauInt << 1
    egInt = egBits.astype(np.int64)

    studentInputs = caloInt | tauInt | egInt
    return studentInputs


def soft_dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = K.cast(y_true, "float32")
    y_pred = K.cast(y_pred, "float32")

    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    sum_true = K.sum(y_true, axis=[1, 2, 3])
    sum_pred = K.sum(y_pred, axis=[1, 2, 3])

    dice_coeff = (2.0 * intersection + smooth) / (sum_true + sum_pred + smooth)
    return 1.0 - dice_coeff


def hybrid_bce_dice_loss(y_true, y_pred):
    binary_crossentropy = BinaryCrossentropy(reduction="none")
    bce_loss = K.mean(binary_crossentropy(y_true, y_pred), axis=[1, 2])

    dice_loss_val = soft_dice_loss(y_true, y_pred)

    return tf.cast(bce_loss, tf.float32) + tf.cast(dice_loss_val, tf.float32)


def loadTeacherModel(filePath):
    return keras.models.load_model(
        filePath, custom_objects={"hybrid_bce_dice_loss": hybrid_bce_dice_loss}
    )


def makeTargets(teacher_model, caloRegions, tauBits, egBits):
    inputs = np.concatenate(
        [
            caloRegions.reshape((-1, 18, 14, 1)),
            tauBits.reshape((-1, 18, 14, 1)),
            egBits.reshape((-1, 18, 14, 1)),
        ],
        axis=3,
    )

    console.log("Making predictions")

    teacherPredictions = teacher_model.predict(inputs)
    lossFn = teacher_model.loss

    energy_outputs = teacherPredictions[0]
    tau_ouputs = teacherPredictions[1]
    eg_outputs = teacherPredictions[2]

    energy_lossFn = lossFn["output_energy"]
    tau_lossFn = lossFn["output_tau"]
    eg_lossFn = lossFn["output_egamma"]

    # loss = np.array(lossFn(inputs, teacherPredictions))

    console.log("Making individual losses")
    energy_loss = np.mean(
        np.array(energy_lossFn(inputs[..., 0:1], energy_outputs)), axis=(1, 2)
    )
    tau_loss = np.array(tau_lossFn(inputs[..., 1:2], tau_ouputs))
    eg_loss = np.array(eg_lossFn(inputs[..., 2:3], eg_outputs))
    console.print(energy_loss.shape)
    console.print(tau_loss.shape)
    console.print(eg_loss.shape)
    console.log("Making final loss and adjustment")
    loss = 0.5 * energy_loss + tau_loss + eg_loss

    adjustedLoss = np.clip(32.0 * np.log(loss + 1e-12), a_min=0.0, a_max=256.0)

    return adjustedLoss


def trainStudentModel(model, inputs, targets, weights=None):
    if weights is None:
        train_inputs, testval_inputs, train_targets, testval_targets = train_test_split(
            inputs, targets, test_size=0.2, random_state=42
        )

        test_inputs, val_inputs, test_targets, val_targets = train_test_split(
            testval_inputs, testval_targets, test_size=0.1 / 0.2, random_state=123
        )

        train_weights, val_weights, test_weights = None, None, None
    else:
        (
            train_inputs,
            testval_inputs,
            train_targets,
            testval_targets,
            train_weights,
            testval_weights,
        ) = train_test_split(inputs, targets, weights, test_size=0.2, random_state=42)

        (
            test_inputs,
            val_inputs,
            test_targets,
            val_targets,
            test_weights,
            val_weights,
        ) = train_test_split(
            testval_inputs,
            testval_targets,
            testval_weights,
            test_size=0.1 / 0.2,
            random_state=123,
        )

    model.fit(
        train_inputs,
        train_targets,
        epochs=1000,
        validation_data=(val_inputs, val_targets, val_weights),
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                "data/3channel_vae_student", save_best_only=True
            ),
            keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=10),
            keras.callbacks.CSVLogger("data/logs/3channel_vae_student_log.csv"),
        ],
        sample_weight=train_weights,
        batch_size=128,
    )

    console.log("Evaluation")
    model.evaluate(test_inputs, test_targets, sample_weight=test_weights)

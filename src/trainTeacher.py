from collections.abc import Callable

import h5py
import numpy as np
import tensorflow as tf
from rich.console import Console
from tensorflow import keras

console = Console()


def loadFile(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path) as theFile:
        etGrid = np.array(theFile["et"])
        tauBitGrid = np.array(theFile["tauBits"])
        egBitGrid = np.array(theFile["egBits"])
    etGrid = etGrid.reshape((-1, 18, 14, 1))
    tauBitGrid = tauBitGrid.reshape((-1, 18, 14, 1))
    egBitGrid = egBitGrid.reshape((-1, 18, 14, 1))
    # return np.concatenate([etGrid, tauBitGrid, egBitGrid], axis=-1)
    return etGrid, tauBitGrid, egBitGrid


def make_mse_bce_loss(alpha: float, beta: float) -> Callable:
    @keras.saving.register_keras_serializable(package="CustomLoss", name="CustomLoss")
    def mse_bse_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        et_true = y_true[:, :, :, 0]
        et_pred = y_pred[:, :, :, 0]

        et_mse = keras.losses.MeanSquaredError()(et_true, et_pred)

        tauBit_true = y_true[:, :, :, 1]
        tauBit_pred = y_pred[:, :, :, 1]

        tauBit_bce = keras.losses.BinaryCrossentropy()(tauBit_true, tauBit_pred)

        egBit_true = y_true[:, :, :, 2]
        egBit_pred = y_pred[:, :, :, 2]

        egBit_bce = keras.losses.BinaryCrossentropy()(egBit_true, egBit_pred)

        return et_mse + alpha * tauBit_bce + beta * egBit_bce

    return mse_bse_loss


def makeTeacherModel(
    inputShape: tuple,
    alpha: int,
    beta: int,
    use3Channels: bool = False,
) -> keras.Model:
    """Encoder"""
    inputLayer = keras.layers.Input(shape=inputShape)
    conv_1 = keras.layers.Conv2D(
        20,
        (3, 3),
        strides=1,
        padding="same",
        name="teacher_conv2d_1",
        kernel_initializer="he_normal",
    )(inputLayer)
    act_1 = keras.layers.Activation("relu", name="teacher_relu_1")(conv_1)
    av_pool_1 = keras.layers.AveragePooling2D(
        (2, 2),
        name="teacher_pool_1",
    )(act_1)
    conv_2 = keras.layers.Conv2D(
        30,
        (3, 3),
        strides=1,
        padding="same",
        name="teacher_conv2d_2",
        kernel_initializer="he_normal",
    )(av_pool_1)
    act_2 = keras.layers.Activation(
        "relu",
        name="teacher_relu_2",
    )(conv_2)
    flat = keras.layers.Flatten(name="teacher_flatten")(act_2)
    dense_1 = keras.layers.Dense(80, activation="relu", name="teacher_latent")(flat)

    """Decoder"""
    dense_2 = keras.layers.Dense(9 * 7 * 30, name="teacher_dense")(dense_1)
    reshape = keras.layers.Reshape(
        (9, 7, 30),
        name="teacher_reshape",
    )(dense_2)
    act_3 = keras.layers.Activation("relu", name="teacher_relu_3")(reshape)
    conv_3 = keras.layers.Conv2D(
        30,
        (3, 3),
        strides=1,
        padding="same",
        name="teacher_conv2d_3",
        kernel_initializer="he_normal",
    )(act_3)
    act_4 = keras.layers.Activation("relu", name="teacher_relu_4")(conv_3)
    conv_transpose = keras.layers.Conv2DTranspose(
        30,
        (3, 3),
        strides=2,
        padding="same",
        name="teacher_conv_transpose",
        activation="relu",
        kernel_initializer="he_normal",
    )(act_4)
    conv_4 = keras.layers.Conv2D(
        20,
        (3, 3),
        strides=1,
        padding="same",
        name="teacher_conv2d_4",
        kernel_initializer="he_normal",
    )(conv_transpose)
    act_5 = keras.layers.Activation("relu", name="teacher_relu_5")(conv_4)

    et_out = keras.layers.Conv2D(
        1,
        (3, 3),
        activation="relu",
        strides=1,
        padding="same",
        name="teacher_et_output",
        kernel_initializer="he_normal",
    )(act_5)

    if use3Channels:
        tauBit_out = keras.layers.Conv2D(
            1,
            (3, 3),
            activation="sigmoid",
            strides=1,
            padding="same",
            name="teacher_tauBit_output",
        )(act_5)

        egBit_out = keras.layers.Conv2D(
            1,
            (3, 3),
            activation="sigmoid",
            strides=1,
            padding="same",
            name="teacher_egBit_output",
        )(act_5)

        concat = keras.layers.Concatenate()([et_out, tauBit_out, egBit_out])
        model = keras.Model(
            inputs=inputLayer,
            outputs=concat,
            # outputs = et_out
        )
        lossFn = make_mse_bce_loss(
            alpha=alpha,
            beta=beta,
        )
        model.compile(optimizer="nadam", loss=lossFn)

    else:
        model = keras.Model(
            inputs=inputLayer,
            # outputs=concat,
            outputs=et_out,
        )
        model.compile(optimizer="nadam", loss="mse")

    return model


def trainModel(model, inputs, test_inputs, use3Channels: bool = False):
    console.rule("Training:")

    model.summary()

    if use3Channels:
        modelName = "teacher_model_3channel"
    else:
        modelName = "teacher_model"

    model.fit(
        x=inputs,
        y=inputs,
        validation_split=0.3,
        epochs=300,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=10),
            keras.callbacks.ModelCheckpoint(
                f"data/{modelName}.keras", store_best_only=True
            ),
            keras.callbacks.CSVLogger(f"data/{modelName}_training_log.csv"),
        ],
    )

    console.rule("Evaluation")
    model.evaluate(
        x=test_inputs,
        y=test_inputs,
    )

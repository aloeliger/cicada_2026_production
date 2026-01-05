import tensorflow as tf
from rich.console import Console
from tensorflow import keras

console = Console()


@keras.saving.register_keras_serializable()
class ConstantGaussianNoise(keras.layers.GaussianNoise):
    def call(self, inputs):
        return super().call(inputs, training=True)


def makeLossFn(latent_space_units):
    def custom_loss(y_true, y_pred):
        ET_loss = tf.reduce_mean(
            keras.losses.MeanSquaredError(reduction="none")(
                y_true[..., 0:1], y_pred[..., 0:1]
            ),
            axis=(1, 2),
        )
        EG_loss = tf.reduce_mean(
            keras.losses.BinaryCrossentropy(reduction="none")(
                y_true[..., 1:2], y_pred[..., 1:2]
            ),
            axis=(1, 2),
        )
        tau_loss = tf.reduce_mean(
            keras.losses.BinaryCrossentropy(reduction="none")(
                y_true[..., 2:3], y_pred[..., 2:3]
            ),
            axis=(1, 2),
        )

        mu = y_pred[:, 0, 0, 3 : 3 + latent_space_units]  # shape (batch_size, latent)
        sigma = y_pred[
            :, 0, 0, 3 + latent_space_units : 3 + 2 * latent_space_units
        ]  # shape(batch_size, latent)
        sigma = tf.clip_by_value(sigma, 1e-4, 1e20)

        kl_divergence = 0.5 * tf.reduce_sum(
            mu**2 + sigma**2 - tf.math.log(sigma**1) - 1, axis=-1
        )

        return ET_loss + EG_loss + tau_loss + kl_divergence

    return custom_loss


def make_VAE_Model(latent_space_units, inputShape):
    inputLayer = keras.layers.Input(shape=inputShape, name="inputLayer")
    # normLayer = keras.layers.LayerNormalization(axis=(1, 2), name="normLayer")(
    normLayer = keras.layers.BatchNormalization(name="normLayer")(inputLayer)

    conv_1 = keras.layers.Conv2D(
        latent_space_units // 4,
        kernel_size=3,
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="conv_1",
    )(normLayer)
    maxPool = keras.layers.MaxPooling2D(2, name="maxPool")(conv_1)
    # denoise = keras.layers.SpatialDropout2D(0.2, name="denoise")(maxPool)
    conv_2 = keras.layers.Conv2D(
        latent_space_units // 2,
        kernel_size=3,
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="conv_2",
    )(maxPool)
    # flat = keras.layers.GlobalMaxPooling2D()(conv_2)
    flat = keras.layers.Flatten(name="flat")(conv_2)
    denoise = keras.layers.Dropout(0.2)(flat)
    latent_space_size = keras.layers.Dense(
        latent_space_units,
        activation="relu",
        kernel_initializer="he_normal",
        name="latent_space_resizer",
    )(flat)

    # assemble the parts of the latent space
    # zerosLayer = keras.layers.Lambda(lambda x: tf.zeros_like(x))(flat)
    zerosLayer = keras.layers.Subtract(name="zerosLayer")(
        [latent_space_size, latent_space_size]
    )
    epsilon = ConstantGaussianNoise(1.0, name="epsilon")(zerosLayer)
    # epsilon = keras.layers.GaussianNoise(1.0)(zerosLayer)
    z_sigma = keras.layers.Dense(
        latent_space_units,
        activation="softplus",
        name="z_sigma",
    )(denoise)
    z_mean = keras.layers.Dense(latent_space_units, name="z_mu")(denoise)

    # do the latent space parameterization trick
    sample = keras.layers.Multiply(name="sigma_sample")([z_sigma, epsilon])
    full_latent = keras.layers.Add(name="full_latent")([z_mean, sample])

    # Decoder
    decode_1 = keras.layers.Dense(
        latent_space_units * 9 * 7,
        activation="relu",
        kernel_initializer="he_normal",
        name="decode_1",
    )(full_latent)
    reshape = keras.layers.Reshape(
        target_shape=(9, 7, latent_space_units), name="decode_reshape"
    )(decode_1)
    drop_1 = keras.layers.SpatialDropout2D(0.2, name="decode_denoise")(reshape)
    conv_up_1 = keras.layers.Conv2DTranspose(
        latent_space_units // 2,
        kernel_size=2,
        strides=2,
        padding="valid",
        activation="relu",
        kernel_initializer="he_normal",
        name="decode_conv_transpose",
    )(drop_1)

    # output_layers
    ET_out = keras.layers.Conv2D(
        1,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="ET_outputs",
    )(conv_up_1)

    EG_out = keras.layers.Conv2D(
        1, kernel_size=3, padding="same", activation="sigmoid", name="EG_outputs"
    )(conv_up_1)

    tau_out = keras.layers.Conv2D(
        1, kernel_size=3, padding="same", activation="sigmoid", name="tau_outputs"
    )(conv_up_1)

    z_mean_repeat = keras.layers.RepeatVector(252, name="mu_repeater")(z_mean)
    z_mean_reshape = keras.layers.Reshape(
        target_shape=(18, 14, latent_space_units), name="mu_repeat_reshape"
    )(z_mean_repeat)

    z_sigma_repeat = keras.layers.RepeatVector(252, name="sigma_repeater")(z_mean)
    z_sigma_reshape = keras.layers.Reshape(
        target_shape=(18, 14, latent_space_units), name="sigma_repeat_reshape"
    )(z_sigma_repeat)

    outputLayer = keras.layers.Concatenate(name="full_output")(
        [ET_out, EG_out, tau_out, z_mean_reshape, z_sigma_reshape]
    )

    model = keras.Model(inputs=inputLayer, outputs=outputLayer)
    lossFn = makeLossFn(latent_space_units)
    model.compile(loss=lossFn, optimizer="nadam")

    return model

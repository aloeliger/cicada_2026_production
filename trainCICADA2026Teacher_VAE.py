import numpy as np
import yaml
from rich.console import Console
from sklearn.model_selection import train_test_split
from tensorflow import keras

from src import teacher_vae, trainTeacher

console = Console()


def main(params):
    console.log("Making VAE model")

    dataPath = params["inputFiles"]["dataFile"]
    caloRegions, tauBits, egBits = trainTeacher.loadFile(dataPath)
    dataGrids = np.concatenate([caloRegions, tauBits, egBits], axis=-1)

    train_grids, test_grids = train_test_split(
        dataGrids, test_size=0.2, random_state=123
    )

    vae_model = teacher_vae.make_VAE_Model(
        latent_space_units=params["cicadaTeacherVAE3Channel"]["latentSpaceSize"],
        inputShape=dataGrids.shape[1:],
        alpha=params["alpha"],
        beta=params["beta"],
    )

    vae_model.summary()

    vae_model.fit(
        x=dataGrids,
        y=dataGrids,
        validation_split=0.3,
        epochs=300,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=10),
            keras.callbacks.ModelCheckpoint(
                "data/teacher_model_vae.keras", store_best_only=True
            ),
            keras.callbacks.CSVLogger("data/training_vae_log.csv"),
        ],
    )

    console.rule("Evaluation")
    vae_model.evaluate(
        x=test_grids,
        y=test_grids,
    )


if __name__ == "__main__":
    with open("params.yaml") as theFile:
        params = yaml.safe_load(theFile)

    main(params)

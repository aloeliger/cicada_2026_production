import numpy as np
import yaml
from rich.console import Console
from tensorflow import keras

from src import encoderOnlyModel as em
from src.teacher_vae import makeLossFn

console = Console()
nprng = np.random.default_rng(123)


def main(params):
    # teacherModelLocation = params["cicadaStudentVAE3Channel"]["teacherModel"]

    loss_fn = makeLossFn(params["cicadaTeacherVAE3Channel"]["latentSpaceSize"])
    teacherModelLocation = params["cicadaEncoderOnly"]["model"]
    console.log("Current Model:")
    # model = csVAE.loadTeacherModel(teacherModelLocation)
    model = keras.model.load_model(
        teacherModelLocation, custom_objects={"lossFn": loss_fn}
    )
    model.summary()

    # encoder = em.getEncoderOnly(
    #     model,
    #     "teacher_inputs_",
    #     "z_mean",
    # )
    # encoder = em.getEncoderFromLayerList(model, params['cicadaEncoderOnly']['layers'])
    console.log("Making encoder only")
    encoder = em.getRebuiltEncoder(model, params["cicadaEncoderOnly"]["layers"])
    console.log("New encoder")
    encoder.summary()
    encoder.save("data/encoderOnlyModel")

    console.log("Random inputs")

    console.log("Making Python Based Usability Model")
    pythonEncoder = em.getEncoderOnly(
        model,
        "teacher_inputs_",
        "z_mean",
    )
    pythonEncoder.save("data/encoderOnlyModel_pythonUsability")

    randomET = nprng.integers(low=0, high=40, size=(1, 18, 14, 1), endpoint=True)
    randomEGBits = nprng.integers(low=0, high=1, size=(1, 18, 14, 1), endpoint=True)
    randomTauBits = nprng.integers(low=0, high=1, size=(1, 18, 14, 1), endpoint=True)

    randomInput = np.stack([randomET, randomEGBits, randomTauBits], axis=3)
    predictions = pythonEncoder.predict(randomInput)
    predictions = np.array(predictions)
    console.log(predictions)


if __name__ == "__main__":
    with open("params.yaml") as theFile:
        params = yaml.safe_load(theFile)

    main(params)

import numpy as np
import yaml
from rich.console import Console

from src import cicadaStudent_vae_3channel as csVAE
from src import encoderOnlyModel as em

console = Console()
nprng = np.random.default_rng(123)


def main(params):
    teacherModelLocation = params["cicadaStudentVAE3Channel"]["teacherModel"]
    console.log("Current Model:")
    model = csVAE.loadTeacherModel(teacherModelLocation)
    model.summary()

    encoder = em.getEncoderOnly(
        model,
        "teacher_inputs_",
        "z_mean",
    )
    # encoder = em.getEncoderFromLayerList(model, params['cicadaEncoderOnly']['layers'])
    console.log("Making encoder only")
    encoder.save("data/encoderOnlyModel")

    console.log("Random inputs")

    randomET = nprng.integers(low=0, high=40, size=(1, 18, 14, 1), endpoint=True)
    randomEGBits = nprng.integers(low=0, high=1, size=(1, 18, 14, 1), endpoint=True)
    randomTauBits = nprng.integers(low=0, high=1, size=(1, 18, 14, 1), endpoint=True)

    randomInput = np.stack([randomET, randomEGBits, randomTauBits], axis=3)
    predictions = encoder.predict(randomInput)
    predictions = np.array(predictions)
    console.log(predictions)


if __name__ == "__main__":
    with open("params.yaml") as theFile:
        params = yaml.safe_load(theFile)

    main(params)

import yaml

from src import cicadaStudent_vae_3channel as csVAE
from src import encoderOnlyModel as em


def main(params):
    teacherModelLocation = params["cicadaStudentVAE3Channel"]["teacherModel"]
    model = csVAE.loadTeacherModel(teacherModelLocation)
    model.summary()

    encoder = em.getEncoderOnly(
        model,
        "teacher_inputs_",
        "z_mean",
    )

    encoder.save("data/encoderOnlyModel")


if __name__ == "__main__":
    with open("params.yaml") as theFile:
        params = yaml.safe_load(theFile)

    main(params)

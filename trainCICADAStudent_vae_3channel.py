import yaml
from rich.console import Console

import src.cicadaStudent_classic as cicadaStudent_classic
import src.cicadaStudent_vae_3channel as cicada_3channel

console = Console()


def main(params):
    console.log("Making CICADA Student: 3 Channel, VAE")
    caloRegions, taubit, egbit, npvs, npvs_good = cicadaStudent_classic.getInputs(
        params
    )

    console.log(f"Calo regions shape: {caloRegions.shape}")
    console.log(f"tau bit shape: {taubit.shape}")
    console.log(f"eg bit shape: {egbit.shape}")

    console.log("Constructing student inputs")
    studentInputs = cicada_3channel.createStudentModelInputs(caloRegions, taubit, egbit)

    teacher_model = cicada_3channel.loadTeacherModel(
        params["cicadaStudentVAE3Channel"]["teacherModel"]
    )
    student_model = cicadaStudent_classic.getModel(studentInputs.shape[1:])

    console.log("Making targets and weights")
    targets = cicada_3channel.makeTargets(
        teacher_model,
        caloRegions,
        taubit,
        egbit,
    )

    weights = cicadaStudent_classic.makeScoreWeights(
        targets, params["cicadaStudentVAE3Channel"]["scoreHistogramOutput"]
    )

    console.log("Training student model")
    cicada_3channel.trainStudentModel(
        student_model, studentInputs, targets, weights=weights
    )


if __name__ == "__main__":
    with open("params.yaml") as theFile:
        params = yaml.safe_load(theFile)
    main(params)

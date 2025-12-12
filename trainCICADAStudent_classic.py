import yaml
from rich.console import Console

import src.cicadaStudent_classic as cicadaStudent_classic
from src import utils

console = Console()


def main(params):
    console.log("Getting list of files")
    fileList = utils.buildFileList(params["cicadaStudentCommon"]["fileDir"])

    console.log("Making CICADA student: classic")
    caloRegions, taubit, egbit, npvs, npvs_good = cicadaStudent_classic.getInputs(
        fileList,
    )

    console.log(f"Calo regions shape: {caloRegions.shape}")
    # console.log(taubit.shape)
    # console.log(egbit.shape)

    teacher_model = cicadaStudent_classic.getTeacherModel(
        params["cicadaStudentClassic"]["teacherModel"]
    )
    student_model = cicadaStudent_classic.getModel(caloRegions.shape[1:])

    console.log("Making targets and weights")
    targets = cicadaStudent_classic.makeTargets(
        teacher_model, caloRegions, taubit, egbit
    )

    weights = cicadaStudent_classic.makeScoreWeights(
        targets, params["cicadaStudentClassic"]["scoreHistogramOutput"]
    )

    console.log("Training student model")
    cicadaStudent_classic.trainStudentModel(
        student_model,
        caloRegions,
        targets,
        weights=weights,
    )


if __name__ == "__main__":
    with open("params.yaml") as theFile:
        params = yaml.safe_load(theFile)
    main(params)

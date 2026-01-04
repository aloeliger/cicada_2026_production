import argparse

import numpy as np
import yaml
from rich.console import Console

import src.cicadaStudent_classic as cicadaStudent_classic
import src.trainTeacher as trainTeacher

console = Console()


def main(args, params):
    console.log("Getting list of files")

    dataPath = params["inputFiles"]["dataFile"]
    caloRegions, tauBits, egBits = trainTeacher.loadFile(dataPath)

    if args.use3Channels:
        dataGrids = np.concatenate([caloRegions, tauBits, egBits], axis=-1)
    else:
        dataGrids = caloRegions

    # fileList = utils.buildFileList(
    #     params["cicadaStudentCommon"]["fileDir"],
    #     params["cicadaStudentCommon"]["dataFileDilation"],
    # )

    console.log("Making CICADA student: classic")
    # caloRegions, taubit, egbit, npvs, npvs_good = cicadaStudent_classic.getInputs(
    #     fileList,
    # )

    console.log(f"Calo regions shape: {caloRegions.shape}")
    # console.log(taubit.shape)
    # console.log(egbit.shape)

    if args.use3Channels:
        studentType = "cicadaStudentClassic_3Channel"
    else:
        studentType = "cicadaStudentClassic"

    console.log(f"Student type: {studentType}")
    teacher_model = cicadaStudent_classic.getTeacherModel(
        params[studentType]["teacherModel"],
        studentType=studentType,
        alpha=params["alpha"],
        beta=params["beta"],
    )
    teacher_model.summary()
    student_model = cicadaStudent_classic.getModel(dataGrids.shape[1:])

    console.log("Making targets and weights")
    targets = cicadaStudent_classic.makeTargets(
        teacher_model,
        dataGrids,
    )

    weights = cicadaStudent_classic.makeScoreWeights(
        targets, params[studentType]["scoreHistogramOutput"]
    )

    console.log(targets.shape)
    console.log(weights.shape)

    console.log("Training student model")
    cicadaStudent_classic.trainStudentModel(
        student_model, dataGrids, targets, weights=weights, studentType=studentType
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--use3Channels",
        action="store_true",
    )

    with open("params.yaml") as theFile:
        params = yaml.safe_load(theFile)

    args = parser.parse_args()

    main(args, params)

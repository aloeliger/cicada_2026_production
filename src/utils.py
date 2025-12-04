import os


def convert_eff_to_rate(eff, nBunches=2544):
    return eff * (float(nBunches) * 11425e-3)


def buildFileList(filePath: str) -> list[str]:
    allFiles = []
    for root, _, files in os.walk(filePath):
        for fileName in files:
            allFiles.append(f"{root}/{fileName}")
    return allFiles

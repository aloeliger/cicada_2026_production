import os
from typing import Optional


def convert_eff_to_rate(eff, nBunches=2544):
    return eff * (float(nBunches) * 11425e-3)


def buildFileList(filePath: str, fileDilation: Optional[int] = None) -> list[str]:
    allFiles = []
    for root, _, files in os.walk(filePath):
        for index, fileName in enumerate(files):
            if fileDilation is not None:
                if index % fileDilation != 0:
                    continue
            allFiles.append(f"{root}/{fileName}")
    return allFiles

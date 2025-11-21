import os


def buildFileList(filePath: str) -> list[str]:
    allFiles = []
    for root, _, files in os.walk(filePath):
        for fileName in files:
            allFiles.append(f"{root}/{fileName}")
    return allFiles

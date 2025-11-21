import time

import h5py
import numpy as np
import ROOT
import uproot
from rich.console import Console
from rich.live import Live

from . import utils

console = Console()


def reshapeInput(
    inputs: np.ndarray,
    ieta: np.ndarray,
    iphi: np.ndarray,
    total_iphi: int = 18,
    total_ieta: int = 14,
) -> np.ndarray:
    rows = np.arange(len(inputs))[:None]
    reshapedInput = np.zeros(shape=(len(inputs), total_iphi, total_ieta))
    reshapedInput[rows, iphi, ieta] = inputs
    return reshapedInput


def processBatch(batch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tauBits = batch.Regions_taubit
    egBits = batch.Regions_egbit
    ets = batch.Regions_et
    iphi = batch.Regions_iphi
    ieta = batch.Regions_ieta

    reshapedModelInput = reshapeInput(
        inputs=ets,
        iphi=iphi,
        ieta=ieta,
    )

    reshapedTauBits = reshapeInput(
        inputs=tauBits,
        iphi=iphi,
        ieta=ieta,
    )

    reshapedEGBits = reshapeInput(inputs=egBits, iphi=iphi, ieta=ieta)

    return reshapedModelInput, reshapedTauBits, reshapedEGBits


def processFiles(
    fileType: str,
    filePath: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Let's first put together all the files that will go into this
    fileList = utils.buildFileList(filePath)
    countChain = ROOT.TChain("Event")
    for fileName in fileList:
        countChain.Add(fileName)
    fileList = [x + ":Event" for x in fileList]
    branchesToLoad = [
        "run",
        "L1_ZeroBias",
        "Regions_taubit",
        "Regions_egbit",
        "Regions_iphi",
        "Regions_ieta",
        "Regions_ieta",
    ]

    def usedBranches(x):
        return x in branchesToLoad

    """Now we uproot iterate through all these files to get the batches.
    When we have a batch, we will process it down into a set of grids"""

    totalEvents = countChain.GetEntries()
    console.log(f"Total Events: {totalEvents}")

    with Live(console=console) as live:
        eventsProcessed = 0
        processedBatches = 0
        startTime = time.perf_counter()
        allEtGrids = []
        allTauBitGrids = []
        allEGBitGrids = []
        for batch in uproot.iterate(
            fileList, filter_name=usedBranches, step_size="1 GB"
        ):
            etGrids, tauBitGrids, egBitGrids = processBatch(batch)
            allEtGrids.append(etGrids)
            allTauBitGrids.append(tauBitGrids)
            allEGBitGrids.append(egBitGrids)
            eventsProcessed += len(etGrids)
            processedBatches += 1
            currentTime = time.perf_counter()
            live.update(
                f"""
                Processed batches: {processedBatches}
                Processed events: {eventsProcessed}
                Elapsed time: {currentTime - startTime:.2g}
                Percent completed: {eventsProcessed / totalEvents:.2%}
                """
            )
    allEtGrids = np.concatenate(allEtGrids, axis=0)
    allTauBitGrids = np.concatenate(allTauBitGrids, axis=0)
    allEGBitGrids = np.concatenate(allEGBitGrids, axis=0)
    return allEtGrids, allTauBitGrids, allEGBitGrids


def saveGrids(
    etGrids: np.ndarray,
    tauBitGrids: np.ndarray,
    egBitGrids: np.ndarray,
    outputPath: str,
) -> None:
    with h5py.File(outputPath, "w") as theFile:
        theFile.create_dataset("et", data=etGrids)
        theFile.create_dataset("tauBits", data=tauBitGrids)
        theFile.create_dataset("egBits", data=egBitGrids)

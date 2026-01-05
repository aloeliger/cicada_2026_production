#!/usr/bin/env python3

import argparse
import pathlib
import shutil


def main(args):
    firmwarePath = pathlib.Path(args.firmwareLocation)
    testingPath = pathlib.Path(args.testingLocation)

    print("Doing some preliminary checks...")

    # Double check the directories and files exist in the places we want them too
    assert (firmwarePath / "cicada.h").exists(), "Couldn't find the CICADA header"
    assert (firmwarePath / "cicada.h").is_file(), "CICADA header is not a file"
    print("Found the source cicada header")

    assert (firmwarePath / "cicada.cpp").exists(), "Couldn't find the CICADA core code"
    assert (firmwarePath / "cicada.cpp").is_file(), "CICADA core code is not a file"
    print("Found the source cicada core code")

    assert (firmwarePath / "defines.h").exists(), "Couldn't find the defines header"
    assert (firmwarePath / "defines.h").is_file(), "defines header is not a file"
    print("Found the defines header")

    assert (firmwarePath / "parameters.h").exists(), (
        "Couldn't find the parameters header"
    )
    assert (firmwarePath / "parameters.h").is_file(), "parameters header is not a file"
    print("Found the parameters header")

    assert (firmwarePath / "weights").exists(), "Couldn't find the weights directory"
    assert (firmwarePath / "weights").is_dir(), "weights directory is not a directory"
    print("Found the weights directory")

    assert (firmwarePath / "nnet_utils").exists(), "Couldn't find the nent utils"
    assert (firmwarePath / "nnet_utils").is_dir(), (
        "nnet utils directory is not a directory"
    )
    print("Found the nnet utils directory")

    # Double check that the files and directories exist
    # to be replaced at the target destination.
    # If they don't we might be copying to the wrong place

    assert (testingPath / "cicada.h").exists(), (
        "Couldn't find the CICADA header, at the target"
    )
    assert (testingPath / "cicada.h").is_file(), (
        "CICADA header is not a file, at the target"
    )
    print("Found the target cicada header")

    assert (testingPath / "cicada.cpp").exists(), (
        "Couldn't find the CICADA core code, at the target"
    )
    assert (testingPath / "cicada.cpp").is_file(), (
        "CICADA core code is not a file, at the target"
    )
    print("Found the target cicada code code")

    assert (testingPath / "defines.h").exists(), (
        "Couldn't find the defines header, at the target"
    )
    assert (testingPath / "defines.h").is_file(), (
        "defines header is not a file, at the target"
    )
    print("Found the target defines")

    assert (testingPath / "parameters.h").exists(), (
        "Couldn't find the parameters header, at the target"
    )
    assert (testingPath / "parameters.h").is_file(), (
        "parameters header is not a file, at the target"
    )
    print("Found the target parameters")

    assert (testingPath / "weights").exists(), (
        "Couldn't find the weights directory, at the target"
    )
    assert (testingPath / "weights").is_dir(), (
        "weights directory is not a directory, at the target"
    )
    print("Found the target weights directory")

    assert (testingPath / "nnet_utils").exists(), (
        "Couldn't find the nent utils, at the target"
    )
    assert (testingPath / "nnet_utils").is_dir(), (
        "nnet utils directory is not a directory, at the target"
    )
    print("Found the target nnet utils directory")

    print()
    print("Copying cicada header...")
    shutil.copyfile(firmwarePath / "cicada.h", testingPath / "cicada.h")

    print("Copying cicada core code...")
    shutil.copyfile(firmwarePath / "cicada.cpp", testingPath / "cicada.cpp")

    print("Copying defines...")
    shutil.copyfile(firmwarePath / "defines.h", testingPath / "defines.h")

    print("Copying parameters...")
    shutil.copyfile(firmwarePath / "parameters.h", testingPath / "parameters.h")

    print("Copying weights directory...")
    shutil.copytree(
        firmwarePath / "weights", testingPath / "weights", dirs_exist_ok=True
    )

    print("Copying the nnet_utils directory...")
    shutil.copytree(
        firmwarePath / "nnet_utils", testingPath / "nnet_utils", dirs_exist_ok=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy firmware to testing location")

    parser.add_argument(
        "firmwareLocation",
        nargs="?",
        help="Location of the directory containing the .cpp/.h, and weights",
        type=str,
    )

    parser.add_argument(
        "testingLocation",
        nargs="?",
        help="Location of the firmware latency testing repository",
    )

    args = parser.parse_args()

    main(args)

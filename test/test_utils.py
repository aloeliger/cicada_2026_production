import cicada_2026_production.src.utils as utils

hdfs_path = "/hdfs/store/user/aloelige/ZeroBias"


def testBuildFileList(mocker):
    filePath = f"{hdfs_path}/Operations_ZeroBias_Run2025C_Run392991_05Jun2025"

    mocker.patch(
        "os.walk",
        return_value=[
            (
                "/hdfs/store/user/aloelige/ZeroBias/Operations_ZeroBias_Run2025C_Run392991_05Jun2025",
                [],
                ["file1.root", "file2.root", "file3.root"],
            ),
        ],
    )

    fullFileList = utils.buildFileList(filePath)

    assert fullFileList != []
    assert fullFileList == [
        f"{hdfs_path}/Operations_ZeroBias_Run2025C_Run392991_05Jun2025/file1.root",
        f"{hdfs_path}/Operations_ZeroBias_Run2025C_Run392991_05Jun2025/file2.root",
        f"{hdfs_path}/Operations_ZeroBias_Run2025C_Run392991_05Jun2025/file3.root",
    ]

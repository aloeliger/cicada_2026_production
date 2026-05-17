cd ctp7cpp/vivado_hls
source /afs/hep.wisc.edu/cms/sw/Xilinx/Vitis_HLS/2021.1/settings64.sh
LD_LIBRARY_PATH="/afs/hep.wisc.edu/home/aloeliger/lib:$LD_LIBRARY_PATH"

export LD_LIBRARY_PATH
vitis_hls -f run_hls.tcl

#! /bin/bash
echo $"Initializing...\n\n"

# CUDA?
echo "Will you use CUDA?"
read -p "Y/y-YES | N/n-NO " cuda
if [[ $cuda =~ Y|y ]]; then GPU="ON"; fi
if [[ $GPU ]]; then
    if [ ! -d "pytorch" ]; then git clone --recursive https://github.com/pytorch/pytorch; else cd pytorch && git submodule sync && git submodule update --init --recursive && cd ..; fi
    cd pytorch
    pip install pyyaml
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
    MACOSX_DEPLOYMENT_TARGET=10.15 CC=clang CXX=clang++ python setup.py install
fi

# Python dependencies
echo "Please confirm that you're in a python virtual environment"
read -p "Y/y-YES | N/n-NO " virtualenv
if [[ $virtualenv =~ Y|y ]]; then continue; else echo "Aborting" && exit 1; fi
pip install -r requirements.txt

# System dependencies
if ! ffmpeg -version &> /dev/null; then echo $"Attempting to install ffmpeg with Brew...\n" brew install ffmpeg; fi
echo $"ffmpeg is installed\n"
echo "Attempting to download model..." && if [ ! -f "model/remasternet.pth.tar" ]; then bash download_model.sh; fi
echo "Model is downloaded"

# Setting up the script
echo "Ready to run script"
read -p "--input (default: example/a-bomb_blast_effects_part.mp4)" input
read -p "--reference_dir (default:example/references/) " reference_dir

# More options
echo $"Other options?\n\nOptions include:\n--disable_colorization | Only perfform restoration with enhancement\n--gpu (Recommended) Defaults to false\n--mindim | minimum edge dimension. Default:320\n"
read -p "" options
if [[ $input == "" ]]; then export input=example/a-bomb_blast_effects_part.mp4; fi
echo "--input $input"
if [[ $reference_dir == "" ]]; then export reference_dir=example/references/; fi
echo "--reference_dir $reference_dir"

# Running script
echo "Running script..."
echo "$ python remaster.py --input $input --reference_dir $reference_dir $options\n"
start_time=date
python3 remaster.py --input $input --reference_dir $reference_dir $options

# Script complete
end_time=date
echo "Total of $elapsed seconds elapsed for process"

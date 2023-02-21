#!/bin/zsh

cd `dirname $0`/../

echo "Current working directory is ${PWD}"
DIRECTORY="${PWD}/venv"

if ! [[ -d "${DIRECTORY}" && ! -L "${DIRECTORY}" ]] ; then
    python3  -m venv "${DIRECTORY}"
fi

source ${DIRECTORY}/venv/bin/activate

pip3 install pandas seaborn customtkinter scikit-learn numpy

echo -n "Would you like to use tensorflow for Neural Network Regression? (Y/n): "
read use_tensorflow

if [[ "$use_tensorflow" = "Y" || "$use_tensorflow" = "y" ]] ; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Installing tensorflow-macos and tensorflow-metal..."
        pip3 install tensorflow-macos tensorflow-metal
    else
        echo "Installing tensorflow for Linux..."
        pip3 install tensorflow
    fi
else
    echo "Please don't forget to comment out the 'import NNRTabView' line and remove NNRTabView from the list of tab view types in the __init__() function from PredictionApp.py to not use tensorflow related tabs!"
fi

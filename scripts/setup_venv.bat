@echo off

pushd %~dp0\..\

if not exist venv\ (
    python -m venv venv\
)
call .\venv\Scripts\activate.bat
pip install pandas seaborn customtkinter scikit-learn numpy

set /p input = "Would you like to use tensorflow for Neural Network Regression? (Y/n): "

set /a result = 0

if "%input%" == "y" set /a result = 1
if "%input%" == "Y" set /a result = 1

if %result% == 1 (
    pip install tensorflow
)
else (
    echo "Please don't forget to comment out the 'import NNRTabView' line and remove NNRTabView from the list of tab view types in the __init__() function from PredictionApp.py to not use tensorflow related tabs!"
)

popd
PAUSE
# DAFD - Design Automation Based on Fluid Dynamics

## Overview
DAFD builds a mathematical model over experimental data of microfludic dorplet generators that we have already collected in order to provide suggestions for chip design parameters. DAFD takes in a desired droplet size and generation rate and outputs a suggestion for the orifice size, aspect ratio, expansion ratio, orfice length, water inlet size, oil inlet size, the flow rate ratio, and the capillary number to produce a droplet generator that realizes your desired rate and size. DAFD also allows users to enter in constraints on the parameters to allow for greater control of the chip design.


## Usage
Once you download the folder, you can move to the "bin" directory and type in "python3 DAFD_GUI.py". A tkinter GUI should open up for you. If you have dependency issues, you can check the requirements.txt file to locate packages you are missing. In the near future, we hope to have virtualenv and binaries released. 


## Parts 

### bin
This folder has the main scripts.

* **DAFD_GUI.py** - This GUI script should be the main entry point for end-users
* **ForwardModelTester.py** - This script performs cross validation over the foward models so we can determine their accuracy.

Note that testing the interpolator is more complicated, as we do not have an exact function that relates the features to the outputs. Testing must be done by actually designing and testing the chips that DAFD suggests.


### core_logic
This folder contains the main logical scripts that describe how the model should be built and how predictions and suggestions should be collected.

* **ForwardModel.py** - This script predicts takes a chip parameter input and outputs the estimated droplet size and generation rate. It does this by utilizing regime prediction and the regression model.
* **InterModel.py** - This script handles the reverse model. It traverses over the forward model using gradient descent to get a set of chip parameters that encode a generator which most closely realizes the desired generation rate and droplet size.
* **RegimeClassifier.py** - This script uses a machine learning classifier to predict if the given chip parameters encode a generator which is in the jetting regime or the dripping regime.
* **Regressor.py** - This script is an adapter for several regressors (all found in the models folder). It can predict some output based on the given training data. 


### helper_scripts
This folder contains useful scripts that many different scripts will require.

* **ModelHelper.py** - This singleton object provides functionality for loading the experimental data, partitioning it, and providing ways to normalize and de-normalize the data.


### models
This folder contains various adapters for several machine learning and mathematical classifiers and regressors. 

* Linear
* Ridge Regression
* Lasso Regression
* Random Forest
* SVR
* Neural Networks


### old_models
These alternative models are now deprecated because of sub-par accuracy, but they are left in this project for completeness.

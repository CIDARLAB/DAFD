# DAFD - Design Automation of Fluid Dynamics

## Overview
DAFD builds a generative model using experimental data of microfludic droplet generators that we have previously collected in order to provide suggestions for chip design parameters. DAFD takes in a desired droplet size and generation rate and outputs a suggestion for the orifice size, aspect ratio, expansion ratio, orifice length, water inlet size, oil inlet size, the flow rate ratio, and the capillary number to produce a droplet generator that realizes your desired rate and size. DAFD also allows users to enter in constraints on the parameters to allow for greater control of the chip design.

You can find out more on our [main website](http://dafdcad.org/neural-net/index.html). If you simply want to use our software, you should use the software on the site. This GitHub page is set up for users who wish to view and edit our source code or who want to set up a local server. 

## Installation
Download the repository from GitHub, create a venv environment, and install the necessary packages. 

```
git clone https://github.com/CIDARLAB/DAFD.git
cd DAFD
python3 -m venv venv/
venv/bin/pip3 install -r requirements.txt
```

The first time you run DAFD, it will train the models for you. To disable this, go to core_logic/Regressor.py and core_logic/RegimeClassifier.py and set "load_model" to false.



## Usage

You can use the DAFD GUI with this command:
```
venv/bin/python3 DAFD_GUI.py
```

DAFD CMD is the command line tool to use DAFD. This is the tool that is called by the PHP script when DAFD is run online. Data is input through a file called cmd_inputs.txt. Here is an example of that fil for using the generative model:

```
CONSTRAINTS
orifice_size=100:125
aspect_ratio=1.5:1.5
regime=1:1
DESIRED_VALS
generation_rate=150
droplet_size=150
``` 

After the CONSTRAINTS header, you can add as many or as few constraints as you desire. You should format them as the constraint_name=lowerbound:upperbound. If you want to constrain the field to a certain value, just use that value as the upper and lower bound. These are the constraint names we currently support:
```
orifice_size
aspect_ratio
expansion_ratio
normalized_orifice_length
normalized_water_inlet
normalized_oil_inlet
flow_rate_ratio
capillary_number
regime
```

If you want to use the forward model, just format the cmd_inputs.txt file as follows:
```
FORWARD
orifice_size=150
aspect_ratio=2.0
expansion_ratio=2.0
normalized_orifice_length=2.0
normalized_water_inlet=3.0
normalized_oil_inlet=2.0
flow_rate_ratio=10.0
capillary_number=0.5

```


You can run DAFD_CMD by `venv/python3 DAFD_CMD.py`

The output will be printed to the command line. The line will be formatted as follows:
```
BEGIN:175.0|2.5357037440564514|4.014100802555123|1.5|2.007055322736064|4.0|13.0|0.06934148775479082|Predicted|46.69509|150.02354|1|3.8595514588789257|4.948142895998623|149.9709581848114|
```

The output line begins with `BEGIN:` and prints the command line with pipe separated values with the following headers:
```
Orifice Size
Aspect Ratio
Expansion Ratio
Normalized Orifice Length
Normalized Water Inlet
Normalized Oil Inlet
Flow Rate Ratio
Capillary Number
Experimental Source
Generation Rate
Droplet Size
Regime
Oil Flow Rate
Water Flow Rate
Inferred Droplet Size
```


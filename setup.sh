#!/bin/bash
apt-get install python3-venv

# Create a new virtual environment
python3 -m venv img_class

# Activate the virtual environment
source img_class/bin/activate

# Install dependencies
pip install -r /work/cds-viz/assignment2-image-classification-Olihaha/requirements.txt

# Run the scripts
python /work/cds-viz/assignment2-image-classification-Olihaha/src/log_reg.py
python /work/cds-viz/assignment2-image-classification-Olihaha/src/NNC.py

#deactivate the venv
deactivate
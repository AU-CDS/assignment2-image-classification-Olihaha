#!/bin/bash

# Create a new virtual environment
python3 -m venv img_class
source img_class/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the scripts
python src/LOGREG_classifier.py
python src/NN_classifier.py

#deactivate the venv
deactivate
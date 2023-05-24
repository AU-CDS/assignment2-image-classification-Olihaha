#!/usr/bin/env bash
source ./env/bin/activate
python src/LOGREG_classifier.py
python src/NN_classifier.py
deactivate
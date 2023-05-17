
# Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks

---
# Introduction and contents
This repository contains a few files.
Out: Foler userd for outputs / results, 
src: folder containing  scripts for performing image classifcation using the Cifar10 dataset with logistic regression and a neural network classifer.
setup.sh: 
assignment_description.md: 



## Data
The cifar10 dataset (https://www.cs.toronto.edu/~kriz/cifar.html) consists of 60K images seperated into 10 classes.

## Models
The src folder contains two Python scripts logreg_classifier.py and NNN_classifier.py which both import, preprocess and performs classification on the data.


The script "lr_classifier.py" employs multinomial logistic regression for image classification, while the "nn_classifier.py" script utilizes a multi-layer Perceptron classifier. 

## script functions
The Python scripts follow a structured pipeline with the following steps:

1. Importing the data required for the classification task and proprocessing it.
2. loads the cifar-10 dataset
3. Fitting the loaded model to the training data
4. Using the trained model to predict the labels for the test data.
5. Generating a classification report that includes evaluation metrics accuracy, precision, recall, and F1-score for the predictions. This report is both printed and saved in the out folder.
6. a confusion matrix is also generated in the out folder for both of the models.

### copy the repository 
git clone XXXXX
make sure to be in correct directory
(cd assignment2-image-classifcation)

### how to replicate
run the setup script provided which does the following : 
1. Creates a virtual environment specifically for the project
2. Activates the venv
3. Installs the required packages from the requiremnets.txt file
4. Runs both the src files. "(logreg_classifier.py and NNN_classifier.py)"
5. Deactivates the venv

## Results
### Classfication report
Upon comparing the classification reports, it appears that the NN-classifier demonstrates slightly better performance with an accuracy of 35%, while the LR-classifier lags behind with an accuracy of 31%. However, it is worth noting that both classifiers' performances are somewhat disappointing especially considering that the cifar10 dataset has 10 different labels meaning that at random we would end up with 10% accuracy. 
interesstingly enough both of our classifiers are terrible at predicting cats, birds, deers and frogs, lets find out why.

### Log_reg results
              precision    recall  f1-score   support

    airplane       0.36      0.35      0.35      1000
  automobile       0.36      0.39      0.38      1000
        bird       0.24      0.32      0.27      1000
         cat       0.23      0.16      0.19      1000
        deer       0.26      0.15      0.19      1000
         dog       0.31      0.30      0.31      1000
        frog       0.29      0.31      0.30      1000
       horse       0.32      0.31      0.32      1000
        ship       0.35      0.39      0.37      1000
       truck       0.37      0.46      0.41      1000

    accuracy                           0.31     10000
   macro avg       0.31      0.31      0.31     10000
weighted avg       0.31      0.31      0.31     10000

### NN RESULT 
              precision    recall  f1-score   support

    airplane       0.38      0.41      0.40      1000
  automobile       0.40      0.49      0.44      1000
        bird       0.26      0.34      0.30      1000
         cat       0.28      0.11      0.16      1000
        deer       0.27      0.26      0.27      1000
         dog       0.33      0.34      0.34      1000
        frog       0.28      0.29      0.28      1000
       horse       0.45      0.39      0.42      1000
        ship       0.44      0.44      0.44      1000
       truck       0.42      0.47      0.44      1000

    accuracy                           0.35     10000
   macro avg       0.35      0.35      0.35     10000
weighted avg       0.35      0.35      0.35     10000

### confusion matrix
When examining the confusion matrixes they both appear with similar traits, and its difficult to notice big differences. Interesstingly enough both moddels struggle heavily when it comes to predicting animals. Our model is great at predicting airplanes and pretty good at predicting automobiles, especially considering how i would imagine it being difficult to tell apart trucks and automobiles.
some funny outliers to notice are 
Cats and dogs being mixed up often, which isnt really that surprising given the somewhat similar appearance.
Birds and deers getting mixed up often, i wouldnt necesarrily expect these to animals to have similar apperances but without examining the dataset closer i cant give any better predictions

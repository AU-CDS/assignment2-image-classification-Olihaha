import numpy as np
import os
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10
import cv2

#load cifar10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
#convert data to greyscale
X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])
X_train_scaled = (X_train_grey)/255.0
X_test_scaled = (X_test_grey)/255.0

#reshape
nsamples, nx, ny = X_train_scaled.shape
X_train_dataset = X_train_scaled.reshape((nsamples,nx*ny))
nsamples, nx, ny = X_test_scaled.shape
X_test_dataset = X_test_scaled.reshape((nsamples,nx*ny))

#log model
clf = LogisticRegression(penalty="none",
                        tol=0.1,
                        verbose=True,
                        solver="saga",
                        multi_class="multinomial").fit(X_train_dataset, y_train)
#prediction on model
y_pred = clf.predict(X_test_dataset)

#produce report
report = classification_report(y_test, 
                               y_pred)

# Save the classification report to a file. 
with open('/work/cds-viz/assignment2-image-classification-Olihaha/out/log_report.txt', 'w') as f:
    f.write(report)

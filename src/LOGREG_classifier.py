import numpy as np
import os
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10
import cv2
import matplotlib.pyplot as plt

# Define class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Convert data to grayscale
X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])
X_train_scaled = X_train_grey / 255.0
X_test_scaled = X_test_grey / 255.0

# Reshape
nsamples, nx, ny = X_train_scaled.shape
X_train_dataset = X_train_scaled.reshape((nsamples, nx * ny))
nsamples, nx, ny = X_test_scaled.shape
X_test_dataset = X_test_scaled.reshape((nsamples, nx * ny))

# Logistic Regression model
clf = LogisticRegression(
    penalty="none",
    tol=0.1,
    verbose=True,
    solver="saga",
    multi_class="multinomial").fit(X_train_dataset, y_train)

# Predictions
y_pred = clf.predict(X_test_dataset)

# Map numerical labels to class names
y_test_names = [class_names[label] for label in y_test.ravel()]
y_pred_names = [class_names[label] for label in y_pred]

# Classification report
report = classification_report(y_test_names, y_pred_names)
print(report)

# Save the classification report to a file
with open('out/log_report.txt', 'w') as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(y_test_names, y_pred_names)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("LOGREG Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Add labels to each cell
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig('out/log_confusion_matrix.png')
plt.show()
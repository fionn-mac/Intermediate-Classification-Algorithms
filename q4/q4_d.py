"""
--------
Course: Statistical Methods in Artificial Intelligence (CSE471)
Semester: Fall '17
Professor: Gandhi, Vineet
--------
"""

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sys import argv

def get_input_data(filename, test):
    """
    Function to read the input data from the letter recognition data file.

    Parameters
    ----------
    filename: The path to input data file

    Returns
    -------
    X: The input for the SVM classifier of the shape [n_samples, n_features].
       n_samples is the number of data points (or samples) that are to be loaded.
       n_features is the length of feature vector for each data point (or sample).
    Y: The labels for each of the input data point (or sample). Shape is [n_samples,].

    """

    X = []; Y = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            if test:
                X.append([float(x) for x in line])
            else:
                Y.append(int(line[-1]))
                X.append([float(x) for x in line[0:-1]])
    X = np.asarray(X); Y = np.asarray(Y)

    return X, Y

def calculate_accuracy(predictions, labels):
    """
    Function to calculate the accuracy for a given set of predictions and
    corresponding labels.

    Parameters
    ----------
    predictions: The predictions obtained as output from the logistic regression classifier
    labels: The true label values corresponding to the entries in predictions

    Returns
    -------
    accuracy: Fraction of total samples that have correct predictions (same as
    true label)

    """

    correct = 0
    wrong = 0

    for i, prediction in enumerate(predictions):
        if prediction > 0.5:
            if labels[i] == 1:
                correct += 1
            else:
                wrong += 1
        else:
            if labels[i] == 1:
                wrong += 1
            else:
                correct += 1

    accuracy_score = float(correct)/float(correct + wrong)

    return accuracy_score

def classifier(X_Train, Y_Train, X_Test):
    """
    Function to fit a logistic regression classifier to the input data.

    Parameters
    ----------
    X: Numpy array with the input data points
    Y: Numpy array with the classes
    -------

    """

    """
    Create a Logistic Regression instance and fit it to the kernelized data.
    ==========================================================================
    """

    # regression = SGDClassifier(alpha=0.01, loss='squared_loss', max_iter=1000000, penalty='none')
    regression = LinearRegression()
    regression.fit(X_Train, Y_Train)

    """
    ==========================================================================
    """

    Y_Predict = regression.predict(X_Test)
    
    for y in Y_Predict:
        label = 0
        if y > 0.5:
            label += 1
        print label

if __name__ == '__main__':

    if len(argv) != 3:
        print "Usage: python q4_a.py [relative/path/to/train/file] [relative/path/to/test/file]"
        exit()

    """
    Get the input data using the provided function. Store the X and Y returned
    as X_data and Y_data. Use filename found above as the input to the function.
    ==========================================================================
    """

    X_Train, Y_Train = get_input_data(argv[1], False)
    X_Test, Y_Test = get_input_data(argv[2], True)

    classifier(X_Train, Y_Train, X_Test)
    
#!/usr/bin/env python
# coding: utf-8

# # This script plots evaluation metrics of EMPO_3 ML models. 
# * Import eval_wrapper(y_true, y_pred) into your script
# # INPUT: y_true, y_pred as list of class labels in string format
# # OUTPUT: Accuracy, Confusion Matrix heatmap, Classification Report heatmap

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
# sklearn package functions
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

def cm(y_true, y_pred, main_bool):
    # Generates a confusion matrix for given y_true and y_pred
    # load label encoder
    if main_bool:
        le = load('model_joblibs/final_labelEncoder.joblib')
    else:
        le = load('../model_joblibs/final_labelEncoder.joblib')
    # get list of labels
    labels = np.array(le.classes_)
    # confusion matrix object
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # plot confusion matrix with labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    # formatting
    disp.plot(cmap=plt.cm.Blues)
    plt.xticks(rotation=90)
    plt.show()


def plot_classification_report(y_true, y_pred):
    # Generates classification report with color gradation for scores
    df = pd.DataFrame(classification_report(y_true, 
                                   y_pred, 
                                   zero_division=0, 
                                   output_dict=True)).T
    # zero_divison hides UndefinedMetricWarning: zero scores are due to labels not appearing in y_pred
    # In these instances, refer to the 'weighted avg'
    df['support'] = df.support.apply(int)
    df = df.style.background_gradient(cmap='Blues',
                             subset=(df.index[:-3],
            df.columns[:df.columns.get_loc('f1-score')+1])).format(precision=4)
    display(df)
    
    
def eval_wrapper(y_true, y_pred, main_bool=False):
    # Prints F1 Score and plots the confusion matrix and classification report for a given y_true and y_pred
    if main_bool:
        le = load('model_joblibs/final_labelEncoder.joblib')
    else:
        le = load('../model_joblibs/final_labelEncoder.joblib')
    print(f"F1 Score", f1_score(le.transform(y_true), le.transform(y_pred), average='weighted'))
    print (f"Confusion Matrix")
    cm(y_true, y_pred, main_bool)
    print(f"Classification Report")
    plot_classification_report(y_true, y_pred)

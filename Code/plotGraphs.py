#!/usr/bin/env python


from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from os import listdir, path
import pandas as pd
from sklearn.metrics import roc_curve,auc

#  Plot from the python files

# 1. Find all the CSV files
filenames = [path.join('logs', filename) for filename in listdir('logs') if filename.endswith('csv')];



for filename in filenames:

    # 2. get the data ROC curve data from all of them 
    result = pd.read_csv(filename);
    [fpr,tpr,threshold] = roc_curve(result["expected_sentiment"],\
                                    result["sentiment"]);
    roc_auc = auc(fpr, tpr)


    # 3. Plot the data
    print("\nPlotting for file {}...".format(filename));
    plt.figure();
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = {})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Algorithm:\n{}'.format(filename[len("logs/"):-len(".csv")]))
    plt.legend(loc="lower right")
    #plt.show();

    # Save the corresponding figure in PNG format
    png_filename = filename[:-len('.csv')]+'.png';
    png_filename = png_filename.replace(" ","_");
    png_filename = png_filename.replace(":","_");
    print("\nsaving to {}....".format(png_filename))
    plt.savefig(png_filename, bbox_inches='tight')

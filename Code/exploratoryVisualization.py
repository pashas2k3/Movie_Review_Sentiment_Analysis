from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from os import listdir, path
import pandas as pd
from sklearn.metrics import roc_curve,auc
from nltk.probability import FreqDist

# Read the data
originalTrainData = pd.read_csv(path.join\
                                (path.dirname(__file__),\
                                 'data', 'labeledTrainData.tsv'),\
                                header=0, \
                                delimiter="\t", quoting=3);
extraWordVecData = pd.read_csv(path.join\
                               (path.dirname(__file__), \
                                'data', 'unlabeledTrainData.tsv'), \
                               header=0, delimiter="\t", \
                               quoting=3)
# Split them into words and use Counter to calculate frequency of words

def wordList(stopWordRemoval):
    clean_train_reviews = [];
    for review in originalTrainData["review"]:
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(review, stopWordRemoval)))
    print"Got information of labeled data..."
    for review in extraWordVecData["review"]:
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(review, stopWordRemoval)));
    print"Got information of unlabeled data..."
    return clean_train_reviews;

print "Plotting Frequency distribution without stop words..."
withStopWordsFreqDist = FreqDist(wordList(True));
withStopWordsFreqDist.plot(5000);
 

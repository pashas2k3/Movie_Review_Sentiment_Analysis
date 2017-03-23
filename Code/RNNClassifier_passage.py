import numpy as np
import pandas as pd
import os 

from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

from passage.models import RNN
from passage.updates import Adadelta
from passage.layers import Embedding, GatedRecurrent, Dense
from passage.preprocessing import Tokenizer

def TrainTestRNN(trainData, testData):
    ''' Adapted from 
    https://github.com/vinhkhuc/kaggle-sentiment-popcorn/blob/master/scripts/passage_nn.py'''


    print("Loading data ...")

    tokenizer = Tokenizer(min_df=10, max_features=100000)
    trX = tokenizer.fit_transform(list(trainData['review']))
    teX = tokenizer.transform(list(testData['review']))

    print("Training ...")
    layers = [
        Embedding(size=256, n_features=tokenizer.n_features),
        GatedRecurrent(size=512, activation='tanh', gate_activation='steeper_sigmoid',
                       init='orthogonal', seq_output=False, p_drop=0.75),
        Dense(size=1, activation='sigmoid', init='orthogonal')
    ]

    model = RNN(layers=layers, cost='bce', updater=Adadelta(lr=0.5))
    trY = np.array(trainData['sentiment']);
    model.fit(trX, trY , n_epochs=10)
    
    print ("Prediction error on training set");
    pr_trX = model.predict(trX).flatten()
    predY_train = np.ones(len(trY));
    predY_train[pr_trX<0.5] = 0;
    print "Training Prediction Accuracy {}".format(accuracy_score(trY, predY_train));
    
    
    # Predicting the probabilities of positive labels
    print("Predicting ...")
    pr_teX = model.predict(teX).flatten()

    predY = np.ones(len(testData["sentiment"]))
    predY[pr_teX < 0.5] = 0

    output = pd.DataFrame(data={"id":testData["id"], "sentiment":predY,\
                                "expected_sentiment":testData["sentiment"]})

    return output;

if __name__ == '__main__':
    # 1. get the data
        # Read the data
    print "Read the data"
    originalTrainData = pd.read_csv(os.path.join\
                                    (os.path.dirname(__file__),\
                        'data', 'labeledTrainData.tsv'),\
                                    header=0, \
                                    delimiter="\t", quoting=3);
    [trainData, testData] = train_test_split(originalTrainData, \
                                             test_size=0.3,\
                                             random_state=42);

    print "Num of reviews for training {}".format(len(trainData))

    output = TrainTestRNN(trainData, testData);

    output.to_csv(os.path.join(os.path.dirname(__file__), 'logs',"RunTestsLog.csv"), index=False, quoting=3)

    print "Training Prediction Accuracy {}".format(accuracy_score(np.array(testData['sentiment']), predY));


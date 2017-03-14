from gensim.models import Word2Vec
import os
from KaggleWord2VecUtility import KaggleWord2VecUtility
import logging
import pandas as pd
from nltk.corpus import stopwords
import nltk.data

def LoadWordVectorModel(model_short_name):
    model_name = os.path.join (os.path.dirname(__file__), 'data',\
                               model_short_name);
    return Word2Vec.load(model_name);


def TrainWordVectorModel(model_short_name, labeledData, unlabeledData):

    model_name = os.path.join (os.path.dirname(__file__), 'data',\
                               model_short_name);

    # early return if the file already exists
    if os.path.isfile(model_name):
        print "Word vector model already exists"
        return;


    # Import the built-in logging module and configure it so that 
    # Word2Vec creates nice output messages
    logging.basicConfig(format= \
                        '%(asctime)s : %(levelname)s : %(message)s',\
                        level=logging.INFO);

    # Load the punkt tokenizer
    print("Loading tokenizers");
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # ****** Split the labeled and unlabeled training sets into clean sentences
    #
    sentences = []  # Initialize an empty list of sentences

    print "Parsing sentences from labeled set"
    for review in labeledData["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

    print "Parsing sentences from unlabeled set"
    for review in unlabeledData["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

    # Set values for various parameters
    # TODO: Explain the rationale for selecting the different parameters
    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print "Training Word2Vec model (will take some time)..."
    model = Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling, seed=1)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # will load the model later once created
    model.save(model_name)
    print "Model saved to "+ model_name

# # This is for interactive testing only
# if __name__ == '__main__':

#     # Read data from files
#     train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )
#     test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )
#     unlabeled_train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', "unlabeledTrainData.tsv"), header=0,  delimiter="\t", quoting=3 )

#     # Verify the number of reviews that were read (100,000 in total)
#     print "Read %d labeled train reviews, %d labeled test reviews, " \
#      "and %d unlabeled reviews\n" % (train["review"].size,
#      test["review"].size, unlabeled_train["review"].size )

#     # ****** Set parameters and train the word2vec model
#     #

#     TrainWordVectorModel("Temp", train, unlabeled_train);
#     model = LoadWordVectorModel("Temp");

#     print model.doesnt_match("man woman child kitchen".split())
#     print model.doesnt_match("france england germany berlin".split())
#     print model.doesnt_match("paris berlin london austria".split())
#     print model.most_similar("man")
#     print model.most_similar("queen")
#     print model.most_similar("awful")



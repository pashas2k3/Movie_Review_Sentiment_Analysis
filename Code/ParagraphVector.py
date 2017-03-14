from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import os
from KaggleWord2VecUtility import KaggleWord2VecUtility
import logging
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import nltk.data
import json
import pickle
from random import shuffle


def LoadReviewVectorModel(model_short_name, DM):
    suffix = "_dbow";
    if(DM):
        suffix = "_dm";

    model_short_name = model_short_name + suffix;
    
    model_name = os.path.join (os.path.dirname(__file__), 'data',\
                               model_short_name);
    return Doc2Vec.load(model_name);

def getCleanLabeledReviews(reviews, labelizedSentences):
    for index, row in reviews.iterrows():
        try:
            clean_reviews = KaggleWord2VecUtility.review_to_wordlist\
                            (row["review"], remove_stopwords= False);
            id_label = row["id"];
            labelizedSentences.append(LabeledSentence\
                                      (clean_reviews, [id_label]));
        except Exception as interrupt:
            print "Exception caught: "
            print row["review"]
            print "Exception Caught for: "
            print row["id"]

    return labelizedSentences;


def TrainReviewVectorModel(model_short_name, labeledData, \
                           unlabeledData):
    model_name = os.path.join (os.path.dirname(__file__), 'data',\
                               model_short_name);

    # early return if the file already exists
    if os.path.isfile(model_name + "_dbow"):
        print "Doc vector model already exists"
        return;

    # Import the built-in logging module and configure it so that 
    # Doc2Vec creates nice output messages
    logging.basicConfig(format= \
                        '%(asctime)s : %(levelname)s : %(message)s',\
                        level=logging.INFO);


    # create cache if the file doesn't already exists
    sentences = []  # Initialize an empty list of sentences

    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    print "Parsing sentences..."
    sentences = getCleanLabeledReviews(labeledData, sentences);
    sentences = getCleanLabeledReviews(unlabeledData, sentences);

    # Set values for various parameters
    # TODO: Explain the rationale for selecting the different parameters
    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print "Training Doc2Vec model (will take some time)..."
    model_dm = Doc2Vec(workers=num_workers, size=num_features, \
                       min_count = min_word_count, \
                       window = context, sample = downsampling, seed=1)
    model_dbow = Doc2Vec(workers=num_workers, size=num_features, \
                         min_count = min_word_count, \
                         window = context, sample = downsampling, \
                         dm = 0, seed=1)
    print "Building vocabulary"
    model_dm.build_vocab(sentences);
    model_dbow.build_vocab(sentences);

    # Doc2vec gives a better response if the same paragraph is 
    # encountered multiple times in random order.
    for epoch in range(10):
        print "Training for epoch:"
        print epoch
        shuffle(sentences)
        model_dm.train(sentences)
        model_dbow.train(sentences)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model_dm.init_sims(replace=True)
    model_dbow.init_sims(replace=True)

    # will load the model later once created
    model_dm.save(model_name+"_dm")
    model_dbow.save(model_name+"_dbow")

    print "Model saved to "+ model_name + "_dm and "+ model_name + "_dbow"


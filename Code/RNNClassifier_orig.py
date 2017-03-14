#Uses RNN to classify 

# Will need Tensorflow to install it
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.cross_validation import train_test_split
from KaggleWord2VecUtility import KaggleWord2VecUtility
from collections import Counter
import time
#===============================================
# MODEL
#===============================================
# Word 2 vector keep it as a future possibility
# too much hassle learning the API

# REFERENCE: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dynamic_rnn.py
# REFERENCE: http://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html
# Traindata and Testdata are expected to be of dimensions 
# [batch_size, n_inputs]  

def build_graph(batch_size, maxSeqLen, stateSize):
    # Parameters
    learning_rate = 0.003
    training_iters = 1000000

    # Network Parameters
    n_hidden = 512 # hidden layer num of features
    n_classes = 2 # positive review or not

    # tf Graph input
    x = tf.placeholder(tf.float32, [batch_size, maxSeqLen, stateSize])
    y = tf.placeholder(tf.int32, [batch_size])#n_classes
    seqlen = tf.placeholder(tf.int32, [batch_size])

    # Define a lstm cell with tensorflow
    gru_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
    #tf.contrib.rnn.GRUCell(n_hidden);
        
    # Get GRU cell output, providing 'sequence_length' will  
    # perform dynamic calculation.
    outputs, _ = tf.nn.dynamic_rnn (gru_cell, x, \
                                    dtype=tf.float32,\
                                    sequence_length=seqlen);

    # keep_prob = tf.constant(1.0);
    # # Add dropout, as the model otherwise quickly overfits
    # rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, 
    # we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we 
    # build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them 
    # in a Tensor
    # and change back dimension to [batch_size, max_seq_len, n_input]
    outputs = tf.pack(outputs)
    outputs = tf.transpose(outputs, perm=[1, 0, 2])

    # # Hack to build the indexing and retrieve the right output.
    # batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * maxSeqLen + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # Linear activation, using outputs computed above
    # Hidden layer
    layer = {'weights':tf.Variable(tf.random_normal\
                                   ([n_hidden, n_classes])),\
             'biases':tf.Variable(tf.random_normal([n_classes]))};

    logits = tf.matmul(outputs, layer['weights']) + layer['biases']
    preds = tf.nn.softmax(logits);

    correct = tf.equal(tf.cast(tf.argmax(preds,1),tf.int32), y)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost);
    
    # accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return {'x':x,'seqlen':seqlen,'y':y,'cost':cost,'train_step':optimizer,'preds':preds,'accuracy':accuracy}


def TrainAndValidateRNNClassifier(train_data, batch_size, maxSeqLen):
    
    stateSize = len(train_data['x'][0][0]);
    print "State Size";
    print stateSize;

    # Create batch size which can divide it evenly or contain the 
    # whole data
    if(len(train_data['seqlen']) < batch_size):
        batch_size = len(train_data['seqlen']);
   
    graph = build_graph(batch_size, maxSeqLen,stateSize);
    
    # Initializing the variables
    init = tf.global_variables_initializer()

    # Split the data into batch_size
    def chunks_gen(data, n):
        return [data[i:i + n] for i in xrange(0, len(data), n)];

    # Launch the graph
    print "Launch the graph"
    with tf.Session() as sess:
        sess.run(init)
        # Keep training until reach max iterations
        # epoch in range(10)
        #TODO: Shuffle the sentences in trainingData 10 times while
        # feeding to the Neural Network
        print batch_size
        list_x = chunks_gen(train_data['x'], batch_size);
        list_y = chunks_gen(train_data['y'], batch_size);
        list_seqlen = chunks_gen(train_data['seqlen'], batch_size);
        print len(list_x)

        # Train the RNN multiple times to get a better response
        # TODO: May have to do the training in random order
        for epoch in range(2):
            batch = 1;
            for batch_x,batch_y, batch_seqlen in zip(list_x,list_y,list_seqlen):
                # Run optimization op (backprop)
                trainingPrediction, trainingAccuracy, _ \
                    = sess.run([graph['preds'], graph['accuracy'], \
                                graph['train_step']], \
                               feed_dict={graph['x']: batch_x, \
                                          graph['y']: batch_y, \
                                          graph['seqlen']: batch_seqlen});
                print "Batch {}  accuracy = {}".format(batch, trainingAccuracy)
                batch += 1;
                print("Optimization Finished!");
            


# Train directly
def TrainTestRNNStandAlone(train, test):   
    def getDataDetails(data):
        # order words by the frequency and pick top max_vocab_size
        print "Cleaned up review of data"
        clean_reviews=[]
        for review in data["review"]:
            word_list = KaggleWord2VecUtility.review_to_wordlist(review, True);
            clean_reviews.append(word_list);

        print "Cleaned up review information for vectorization"
        flattened_review_word_list = [item for sublist in clean_reviews for item in sublist];
        orig_sorted_word_list = Counter(flattened_review_word_list);
        max_vocab_size = 5000; # Translate the review to a vocabulary- same as max features in bag of words
        sorted_word_list = orig_sorted_word_list.most_common(max_vocab_size);

        # Any word you have can be converted to a unique index 
        # according to ranking in sorted_counted_list
        # Convert the list of words to indexes
        dictionary = {};
        a = 0;
        for key in sorted_word_list:
            dictionary[key[0]] = a;
            a += 1;

        # NOTE: Can use tensorflow tf.one_hot for one hot conversion  
        def one_hot(word):
            temp = np.zeros(max_vocab_size);
            idx = dictionary[word];
            temp[idx] = 1;
            return temp;

        # Create an array and from that a tf.one_hot
        print "Convert to one hot vectors"
        id_data = [];
        seqlen = [];
        for review in clean_reviews:
            review_word_to_id = [];
            for word in review:
                if(dictionary.has_key(word)):
                    review_word_to_id.append(one_hot(word));
            seqlen.append(len(review));
            id_data.append(review_word_to_id);

        
        max_seq_len = max(seqlen);

        # Pad each sentence  with zeros for max_seq_len at the end
        # NOTE: Can use tensorflow tf.train_batch for padding
        id_data_padded = [];
        template = np.zeros(max_vocab_size+1);
        for review in id_data:
            temp = review;
            len_tensor = len(review);
            for _ in range(max_seq_len-len_tensor):
                temp.append(template);
            id_data_padded.append(temp);

        sentiments = data['sentiment'];
        # sentiments = [];
        # for sentiment in data['sentiment']:
        #     sentiments.append([sentiment]);
            
        return max_seq_len, {'x': id_data_padded, \
                             'y': np.array(sentiments),\
                             'seqlen': seqlen}
    
    print "Training data cleanup"
    max_train_seq_len, padded_train_data = getDataDetails(train);
    # max_test_seq_len, test_data = getDataDetails(testData);
        
    # Train the Neural Networks
    print "Create the network"
    print "Length of cleaned up training data {}".format(len(padded_train_data['seqlen']));
    model = TrainAndValidateRNNClassifier(padded_train_data, \
                                          500, max_train_seq_len);
    # result = 

    # output = pd.DataFrame( data={"id":test["id"], "sentiment":result, \
        #                              "expected_sentiment": \
        #                              test["sentiment"]} );


   


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
                                             test_size=0.999,\
                                             random_state=42);

    print "Num of reviews for training {}".format(len(trainData))

    TrainTestRNNStandAlone(trainData, testData);
    # 2. Get all the constructor arguments and create instance


    # 3. Test in 10 iterations
    #     4. Train 100 times for each iteration


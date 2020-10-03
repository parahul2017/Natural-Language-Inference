import json
import collections
import argparse
import random

from util import *

random.seed(42)

def extract_unigram_features(ex):
    """Return unigrams in the hypothesis and the premise.
    Parameters:
        ex : dict
            Keys are gold_label (int, optional), sentence1 (list), and sentence2 (list)
    Returns:
        A dictionary of BoW features of x.
    Example:
        "I love it", "I hate it" --> {"I":2, "it":2, "hate":1, "love":1}
    """
    # BEGIN_YOUR_CODE
    #remove to remove this
    
    import string
    bow={}
    punc = str.maketrans('', '', string.punctuation)
    sentence = ex["sentence1"]
    features = [word.translate(punc) for word in sentence]
    features = [b.lower() for b in features]
    for item in features:
        if item in bow:
            bow[item] = bow[item] + 1
        else:
            bow[item] = 1
    sentence = ex["sentence2"]
    features = [word.translate(punc) for word in sentence]
    features = [b.lower() for b in features]
    for item in features:
        if item in bow:
            bow[item] = bow[item] + 1
        else:
            bow[item] = 1
    #print(bow)
    return bow
    raise Exception
    # END_YOUR_CODE

def extract_custom_features(ex):
    """Design your own features.
    """
    # BEGIN_YOUR_CODE
    #remove to remove this
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
    import string
    bow={}
    punc = str.maketrans('', '', string.punctuation)
    sentence = ex["sentence1"]
    features = [word.translate(punc) for word in sentence]
    features = [b.lower() for b in features]
    for item in features:
        if item in bow:
            if item not in stop_words:
                bow[item] = bow[item] + 1
        elif item not in stop_words:
            bow[item] = 1
    sentence = ex["sentence2"]
    features = [word.translate(punc) for word in sentence]
    features = [b.lower() for b in features]
    for item in features:
        if item in bow:
            bow[item] = bow[item] + 1
        else:
            bow[item] = 1
    return bow
    raise Exception
    # END_YOUR_CODE


def learn_predictor(train_data, valid_data, feature_extractor, learning_rate, num_epochs):
    """Running SGD on training examples using the logistic loss.
    You may want to evaluate the error on training and dev example after each epoch.
    Take a look at the functions predict and evaluate_predictor in util.py,
    which will be useful for your implementation.
    Parameters:
        train_data : [{gold_label: {0,1}, sentence1: [str], sentence2: [str]}]
        valid_data : same as train_data
        feature_extractor : function
            data (dict) --> feature vector (dict)
        learning_rate : float
        num_epochs : int
    Returns:
        weights : dict
            feature name (str) : weight (float)
    """
    # BEGIN_YOUR_CODE
    #remove to remove this
    
        
    import random
    wt = collections.defaultdict(float)
    temp = 1
    for t in range(num_epochs):
        for data in train_data:
            
            dot_product = dot(wt, feature_extractor(data))
            gradient = ((1. / (1. + math.exp(-dot_product))) - data["gold_label"]) if -100. < dot_product else (0. - data["gold_label"])
            for x in feature_extractor(data):
                wt[x] -= (learning_rate / (1 + temp / float(len(train_data)))) * gradient
    return wt
    raise Exception
    # END_YOUR_CODE

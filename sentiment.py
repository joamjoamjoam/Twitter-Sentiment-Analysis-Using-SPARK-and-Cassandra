from __future__ import print_function
import json
import re
import string
import numpy as np

from pyspark import SparkContext, SparkConf
from pyspark import SQLContext
from pyspark.mllib.feature import Normalizer
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils


def tokenize(text):
    tokens = []
    text = text.encode('ascii', 'ignore')  # to decode
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
                  text)  # to replace url with ''
    text = remove_spl_char_regex.sub(" ", text)  # Remove special characters
    text = text.lower()
                  
    for word in text.split():
       if word not in stopwords \
          and word not in string.punctuation \
          and len(word) > 1 \
          and word != '``':
             tokens.append(word)
    return tokens


def doc2vec(document):
    #100D word vector
    doc_vec = np.zeros(100)
    tot_words = 0
    
    for word in document:
        try:
            # get word's features
            vec = np.array(lookup_bd.value.get(word)) + 1
            # print(vec)
            # if features exists in pretrained models, add it to vector
            if vec != None:
                doc_vec += vec
                tot_words += 1
        except:
            continue

    vec = doc_vec / float(tot_words)
    return vec


conf = SparkConf().setAppName("sentiment_analysis")
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")
sqlContext = SQLContext(sc)

#remove stop words, signs and urls
#regex to remove special characters
remove_spl_char_regex = re.compile('[%s]' % re.escape(string.punctuation))
stopwords = [u'rt', u're', u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your',
    u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers',
    u'herself', u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what',
    u'which', u'who', u'whom', u'this', u'that', u'these', u'those', u'am', u'is', u'are', u'was', u'were',
    u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a',
    u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by',
    u'for', u'with', u'about', u'against', u'between', u'into', u'through', u'during', u'before', u'after',
    u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under',
    u'again', u'further', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how', u'all',
    u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'no', u'nor', u'not',
    u'only', u'own', u'same', u'so', u'than', u'too', u'very', u's', u't', u'can', u'will', u'just', u'don',
    u'should', u'now']
#import word2VecM model for feature extracting
lookup = sqlContext.read.parquet("/home/word2vecM_simple/data").alias("lookup")
lookup.printSchema()
lookup_bd = sc.broadcast(lookup.rdd.collectAsMap())

    
#load labeled data for sentiment analysis    
with open('tweets.json', 'r') as f:
    rawTrn_data = json.load(f)
    f.close()
    
trn_data = []
for obj in rawTrn_data['results']:
    token_text = tokenize(obj['text']) #nomalized the tweets text
    tweet_text = doc2vec(token_text) #convert text to vector
    #use LabeledPoint to give each tweets a sentiment label
    trn_data.append(LabeledPoint(obj['polarity'], tweet_text))
    
trnData = sc.parallelize(trn_data)
#print(trnData)


##################  Here to load test data from Cassandra #########

with open(sys.argv[1]) as f:
    rawTst_data = json.load(f)
    f.close()
    
tst_data = []
for obj in rawTst_data['text']:
    token_text = tokenize(obj)
    tweet_text = doc2vec(token_text)
    
tst_dataRDD = sc.parallelize(tst_data)

#train classificatin model
model = RandomForest.trainClassifier(trnData, numClasses=3, categoricalFeaturesInfo = {},numTrees = 3,featureSubsetStrategy = 'auto',
impurity = 'gini', maxDepth = 4, maxBins = 32)

#get prediction of each tweets
predictions = model.predict(tst_dataRDD.map(lambda x: x.features))

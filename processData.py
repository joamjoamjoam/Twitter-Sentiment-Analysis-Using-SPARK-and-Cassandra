from __future__ import print_function

import sys, timeit

from pyspark.sql import SQLContext 
from pyspark import SparkConf
from pyspark import SparkContext
from datetime import datetime
import uuid
import random
from pyspark.sql.types import * 
from pyspark_cassandra import CassandraSparkContext
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
from geopy.geocoders import Nominatim



geolocater = Nominatim()
analyzer = SentimentIntensityAnalyzer()

def setEmotion(tweettext):
   if tweettext is not None:
      if analyzer.polarity_scores(tweettext)["compound"] > .2:
         return 'positive' 
      elif analyzer.polarity_scores(tweettext)["compound"] < -.2:
         return 'negative' 
      else:
         return 'neutral'
   else:
      return 'neutral'

def setLocation(latitude, longitude):
   tmp = '{0}, {1}'.format(latitude,longitude)
   if latitude is not None:
      return geolocater.reverse(tmp) 
   else:
      return 'No Location'


if __name__ == "__main__":
    begin = timeit.default_timer()
    if len(sys.argv) < 2:
        print("Usage: station.py <file>", file=sys.stderr)
        exit(-1)

    conf = SparkConf().setAppName("Twitter Sentiment Nodes: 1 mediumdata")
    sc = CassandraSparkContext(conf=conf) 
    sqlContext = SQLContext(sc)
    for i in xrange(1,len(sys.argv)):

       tweets = sqlContext.read.json(sys.argv[i])
       print('Filename = %s' % sys.argv[i],file=sys.stderr)
       tweets.printSchema()
       tweets.registerTempTable('tweets')

       temp = tweets.map(lambda row: {    'text': row.text if row.text is not None else 'no text',
                                          'name': row.name if row.name is not None else 'no name',
                                          'recordid': str(uuid.uuid1()),
                                          'longitude': str(row.location[0][0][1]) if row.location is not None else 0.00,
                                          'latitude': str(row.location[0][0][0]) if row.location is not None else 0.00,
                                          'date' : row.created_at if row.created_at is not None else 'invalid date',
                                          'emotionscore': analyzer.polarity_scores(row.text)["compound"] if row.text is not None else 0.0,
                                          'emotion': setEmotion(row.text)if row.text is not None else 'neutral'}).collect()
                                        #'state': row.STATE}).collect() 
                                        #'lat': row.LAT,
                                        #'lon': row.LON,
                                        #'elev': row.ELEV}).collect() 

       sc.parallelize(temp).saveToCassandra(keyspace='project', table='tweets')
       sc.parallelize(temp).saveToCassandra(keyspace='project', table='shen')
    end = timeit.default_timer()
    print("Total Process Time = %f seconds" % (end - begin), file=sys.stderr)
    

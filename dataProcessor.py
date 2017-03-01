from __future__ import print_function

import sys

from pyspark.sql import SQLContext 
from pyspark import SparkConf
from pyspark import SparkContext
from datetime import datetime
import uuid
from pyspark.sql.types import * 
from pyspark_cassandra import CassandraSparkContext

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: station.py <file>", file=sys.stderr)
        exit(-1)

    conf = SparkConf().setAppName("Twitter Sentiment")
    sc = CassandraSparkContext(conf=conf) 
    sqlContext = SQLContext(sc)
    
    for i in xrange(1,len(sys.argv)):

       tweets = sqlContext.read.json(sys.argv[i])
       print('Filename = %s' % sys.argv[i],file=sys.stderr)
       tweets.printSchema()
       tweets.registerTempTable('tweets')

       temp = tweets.map(lambda row: {    'text': row.text,
                                          'recordid': str(uuid.uuid1())}).collect()
                                        #'state': row.STATE}).collect()
                                        #'lat': row.LAT,
                                        #'lon': row.LON,
                                        #'elev': row.ELEV}).collect() 

       sc.parallelize(temp).saveToCassandra(keyspace='project', table='tweets')

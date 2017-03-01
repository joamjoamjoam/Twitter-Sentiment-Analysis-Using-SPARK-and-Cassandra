from __future__ import print_function

import sys

from pyspark.sql import SQLContext 
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql.types import * 
from pyspark_cassandra import CassandraSparkContext 


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: station.py <file>", file=sys.stderr)
        exit(-1)

    conf = SparkConf().setAppName("Weather Stations")
    sc = CassandraSparkContext(conf=conf) 
    sqlContext = SQLContext(sc)

    customSchema = StructType([ \
            StructField("USAF", StringType(), True), \
            StructField("WBAN", StringType(), True), \
            StructField("STATION_NAME", StringType(), True), \
            StructField("CTRY", StringType(), True), \
            StructField("STATE", StringType(), True), \
            StructField("LAT", FloatType(), True), \
            StructField("LON", FloatType(), True), \
            StructField("ELEV", FloatType(), True), \
            StructField("BEGIN", StringType(), True), \
            StructField("END", StringType(), True)]) 

    stations = sqlContext.read.format('com.databricks.spark.csv') \
                .options(header='true') \
                .load(sys.argv[1], schema=customSchema)

    us_stations = stations.filter(stations.CTRY == 'US') \
                          .filter(stations.STATE == 'CA') \
                          .filter(stations.ELEV > 3000) \
                          .filter(stations.STATION_NAME != '') 

    print('# of satisfied stations: {}'.format(us_stations.count()))
    temp = us_stations.map(lambda row: {'name': row.STATION_NAME,
                                        'ctry': row.CTRY,
                                        'state': row.STATE,
                                        'lat': row.LAT,
                                        'lon': row.LON,
                                        'elev': row.ELEV}).collect() 

    sc.parallelize(temp).saveToCassandra(keyspace='weather', table='stations')

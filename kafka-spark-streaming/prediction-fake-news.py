import findspark
from numpy import double, fromstring
from pyspark.sql.dataframe import DataFrame
findspark.init()

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
# from pyspark.streaming.kafka import KafkaUtils
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType, StructField, StructType
from pyspark.sql.functions import col, from_csv, from_json, lit
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, IndexToString, Normalizer, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder
import random
import time

## Importing the Dependencies
import numpy as np
import pandas as pd
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


kafka_topic_name = "topic.collection.news"
kafka_prediction_topic = "topic.prediction.news"
kafka_bootstrap_servers = '52.14.46.226:9094'

# Stemming
stem = PorterStemmer() # basically creating an object for stemming! Stemming is basically getting the root word, for eg: loved --> love! 
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ',content) # this basically replaces everything other than lower a-z & upper A-Z with a ' ', for eg apple,bananna --> apple bananna
    stemmed_content = stemmed_content.lower() # to make all text lower case
    stemmed_content = stemmed_content.split() # this basically splits the line into words with delimiter as ' '
    stemmed_content = [stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')] # basically remove all the stopwords and apply stemming to the final data
    stemmed_content = ' '.join(stemmed_content) # this basically joins back and returns the cleaned sentence
    return stemmed_content

vectorizer = CountVectorizer()
def training_model():
    # download stopwords and wordnet
    # nltk.download('stopwords')
    # nltk.download('wordnet')

    # loading the dataset to dataframe
    news_dataset = pd.read_csv('datasets/train.csv')

    # Pre-processing data
    news_dataset = news_dataset.fillna('')
    news_dataset['content'] = news_dataset['author'] + " " + news_dataset['title']
    news_dataset['content'] = news_dataset['content'].apply(stemming)
    X = news_dataset['content'].values
    y = news_dataset['label'].values
    X = vectorizer.fit_transform(X)

    # training the logistic regression model
    model = LogisticRegression(C = 100, penalty = 'l2', solver= 'newton-cg')
    model.fit(X, y)
    return model

def get_prediction(trained_model, processed_news):
    prediction = trained_model.predict(processed_news)
    return "REAL" if (prediction[0]==0) else "FAKE"

def process_received_news(batch_df: DataFrame, batch_id):
    if (batch_df.count() > 0):
        data_frame = batch_df.toPandas()
        print('======================> Data Streaming <======================')
        batch_df.show()
        data_frame['content'] = data_frame['author'] + " " + data_frame['title']
        data_frame['content'] = data_frame['content'].apply(stemming)
        X_test = data_frame['content']
        X_test = vectorizer.transform(X_test)
        predict_value = get_prediction(model, X_test)
        batch_df.withColumn("Label", lit(predict_value))
        batch_df.selectExpr("CAST(id AS STRING) AS key", "CAST(title AS STRING) AS value") \
                .write.format("kafka").option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
                                    .option("topic", kafka_prediction_topic).save()


sc = SparkContext(appName="app-publish-news", master="local[*]")
spark = SparkSession(sc)
# spark = SparkSession.builder.appName("app-publish-news").master("local[*]").getOrCreate()

# Construct a streaming DataFrame that reads from topic
df_message_consume = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
    .option("subscribe", kafka_topic_name) \
    .option("startingOffsets", "latest") \
    .load()

df_value = df_message_consume.selectExpr("CAST(value AS STRING)")
json_schemas = StructType([StructField("id", StringType()), StructField("title", StringType()), \
                            StructField("author", StringType()), StructField("text", StringType())])
df_news = df_value.select(from_json(col("value"),json_schemas).alias("data_news")).select("data_news.*")

# Processing news data and predict fake/real
model = training_model()

df_news.writeStream.foreachBatch(process_received_news).outputMode("update").format("console").start().awaitTermination()
# df_write_stream = df_news.writeStream.foreachBatch(process_received_news).format("console").outputMode("append").start()
# df_write_stream.awaitTermination()

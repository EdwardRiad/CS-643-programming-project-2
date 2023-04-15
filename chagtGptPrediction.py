 # prediction_application.py
import sys
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Wine Quality Prediction - Prediction") \
    .getOrCreate()

# Load the test dataset
test_data_file = sys.argv[1]  # The file path is passed as a command-line argument
test_data = spark.read.csv(test_data_file, header=True, inferSchema=True)

#


 from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark import SparkConf, SparkContext, SQLContext




# Initialize Spark session
spark = SparkSession.builder \
    .appName("Wine Quality Prediction - Training") \
    .getOrCreate()

# Load the training and validation datasets
training_data = spark.read.csv("s3://progproj2-s3/TrainingDataset.csv", header=True, inferSchema=True)
validation_data = spark.read.csv("s3://progproj2-s3/ValidationDataset.csv", header=True, inferSchema=True)

# Preprocess the data
feature_columns = training_data.columns[:-1]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
training_data = assembler.transform(training_data)
validation_data = assembler.transform(validation_data)

# Train the logistic regression model
logistic_regression = LogisticRegression(featuresCol="features", labelCol="quality", maxIter=10)
lr_model = logistic_regression.fit(training_data)

# Save the model to an Amazon S3 bucket
lr_model.save("s3://progproj2-s3/wine_quality_prediction_model")

# Stop the Spark session
spark.stop()



from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import col
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


conf = (SparkConf().setAppName("Train wine app"))
sc = SparkContext("local", conf=conf)
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)

dataFrame = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true', sep=';').load('s3://progproj2-s3/TrainingDataset.csv')
validatedataFrame = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true', sep=';').load('s3://progproj2-s3/ValidationDataset.csv')

# dropping quality column
newDf = dataFrame.select(dataFrame.columns[:11])

outputRdd = dataFrame.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[:11])))

model = RandomForest.trainClassifier(outputRdd,numClasses=10,categoricalFeaturesInfo={}, numTrees=60, maxBins=32, maxDepth=4, seed=42)


validationOutputRdd = validatedataFrame.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[:11])))

predictions = model.predict(validationOutputRdd.map(lambda x: x.features))
labelsAndPredictions = validationOutputRdd.map(lambda lp: lp.label).zip(predictions)

metrics = MulticlassMetrics(labelsAndPredictions)


labelsAndPredictions_df = labelsAndPredictions.toDF()
labelpred = labelsAndPredictions.toDF(["label", "Prediction"])
labelpred.show(truncate=False)
labelpred_df = labelpred.toPandas()

F1score = f1_score(labelpred_df['label'], labelpred_df['Prediction'], average='micro')
print("The f1 score is : ", F1score)
print(confusion_matrix(labelpred_df['label'],labelpred_df['Prediction']))
print(classification_report(labelpred_df['label'],labelpred_df['Prediction']))
print("Accuracy : " , accuracy_score(labelpred_df['label'], labelpred_df['Prediction']))


print("\n\n==== Saving model ====")
#Saving model
model.save(sc, "s3://progproj2-s3/cs643-trainingmodel.model")

print("Model Saved successfully")

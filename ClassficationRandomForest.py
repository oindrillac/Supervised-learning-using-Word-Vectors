import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import sys
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def getRDD(file,sc):
    lines = sc.textFile(file)
    linesHeader = lines.first()
    header = sc.parallelize([linesHeader])
    linesWithOutHeader = lines.subtract(header)
    myRDD = linesWithOutHeader.map(lambda x: x.split('\t'))
    return myRDD

stops = set(stopwords.words("english"))
def review_to_words(raw_review, stops):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)

    # Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    
    # Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    
    # Convert to lower case, split into individual words
    words = letters_only.lower().split()                             

                      
    
    # Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    
    # Join the words back into one string separated by space and return the result.
    return( " ".join( meaningful_words )) 

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Correct Usage: file.py <dataset>  ", file=sys.stderr)
        exit(-1)
    #Creating the spark session
    spark = SparkSession \
    .builder \
    .appName("Counting_Articles") \
    .getOrCreate()

    #Creating the spark context
    sc = spark.sparkContext

    # file name = "labeledTrainData.tsv"
    raw_data = sys.argv[1]

    # Read the train data as RDD
    moviereviews = getRDD(raw_data,sc)

    # Preprocessing the training data to remove stopwords and lemmatize
    moviereviews_words = moviereviews.map(lambda x: [x[0], int(x[1]), review_to_words(x[2],stops)])

    schema = StructType([
        StructField('id', StringType(), True),
        StructField('sentiment', IntegerType(), True),
        StructField('review', StringType(), True),
    ])

    # RDD to Spark DataFrame
    sparkDF = spark.createDataFrame(moviereviews_words,schema)

    #Spark DataFrame to Pandas DataFrame
    moviereviews_df = sparkDF.toPandas()

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.  
    vectorizer = CountVectorizer(analyzer = "word",   \
                                tokenizer = None,    \
                                preprocessor = None, \
                                stop_words = None,   \
                                max_features = 25000) 

    # Transform training data into feature vectors
    data_features = vectorizer.fit_transform(moviereviews_df["review"])

    # Convert the result to numpy array 
    data_features = data_features.toarray()

    # Dividing data into attributes and labels
    X = data_features
    y = moviereviews_df.iloc[:,1].values

    # Splitting data into training set and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

    # Define RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=150)
    clf.fit(X_train, y_train)

    # Getting feature weights by importance
    importances = clf.feature_importances_

    # Get top 20,000 features
    top_idx = np.argsort(importances)[-20000:]

    # Get training set out of top 20000 features indices
    X_train_reduced_features = X_train[:,top_idx]

    # Get testing set out of top 20000 features indices
    X_test_reduced_features = X_test[:,top_idx]

    # Run model on original train and test set
    clf = RandomForestClassifier(n_estimators=150)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test,y_pred)

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Negative','Positive']
    plt.title('Confusion Matrix - Test Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()

    print("Classification report\n",classification_report(y_test,y_pred))

    print("Accuracy score of model with original features: ", accuracy_score(y_test,y_pred))

    # Now run the model on the reduced train and test set
    
    clf_reduced = RandomForestClassifier(n_estimators=150)
    clf_reduced.fit(X_train_reduced_features, y_train)

    # Use the random forest to make sentiment label predictions
    y_pred_reduced = clf_reduced.predict(X_test_reduced_features)

    cm = confusion_matrix(y_test,y_pred_reduced)

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Negative','Positive']
    plt.title('Confusion Matrix - Test Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()

    print("Classification report\n",classification_report(y_test,y_pred_reduced))

    print("Accuracy score of model with reduced features: " ,accuracy_score(y_test,y_pred_reduced))

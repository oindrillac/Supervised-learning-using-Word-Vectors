# Supervised-learning-using-Word-Vectors


## Data Set and Problem Statement

I performed analysis on on a large Kaggle dataset of movie reviews from IMDB to predict whether a movie review is positive or negative. The data is split into training and testing sets. I have used a Random Forest classifier to fit the "Bag of Words" model. This model is further used to predict the sentiment label of movie reviews in the test dataset. I have understood how learning word vectors could play an important role in predictions using supervised learning in a text corpus.

## Methodology

* Pre-processing : Cleaning, Lowercase, Tokenization, Stopwords removal
* Feature Extraction : Bag of Words - CountVectorizer
* Classification : Random Forest
* Dimension Reduction : Feature importances


## Result


Testing the model on the dataset with 25000 features yields about 85.3% accuracy. The figure below shows the precision, recall, f1-score values on the test data.


<img src="https://github.com/oindrillac/Supervised-learning-using-Word-Vectors/blob/master/old.jpg" height="100" title="Old">

The diagram below shows the confusion matrix obtained.

<img src="https://github.com/oindrillac/Supervised-learning-using-Word-Vectors/blob/master/cm1.jpg" width="480" height="300"  title="cm1">

After reducing the dimension and bringing down the features to 20000, the model yields about 86% accuracy. The figure below shows the precision, recall, f1-score values on the test data.

<img src="https://github.com/oindrillac/Supervised-learning-using-Word-Vectors/blob/master/reduced.jpg" width="500" height="100" title="Reduced">

The diagram below shows the confusion matrix obtained.

<img src="https://github.com/oindrillac/Supervised-learning-using-Word-Vectors/blob/master/cm2.jpg" width="340" height="300" title="cm2">

# How to run  

Run the project code by submitting it to spark.


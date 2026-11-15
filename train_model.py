from data import X_train,X_test,y_train,y_test,X,y,yelp_class
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

import nltk
nltk.download('stopwords')

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

nb.fit(X_train,y_train)

#Prediction and Evaluation
predictions = nb.predict(X_test)

#Confusion matrix and Classification report
print(confusion_matrix(y_test,predictions))
print("/n")
print(classification_report(y_test,predictions))

#Create Pipeline
pipe = Pipeline([('bow',CountVectorizer()),
                 ('tfidf',TfidfTransformer()),
                 ('model',MultinomialNB())])

#Redo train_test_split
X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

pipe.fit(X_train,y_train)

#New Predictions and Evaluations
predictions = pipe.predict(X_test)

#New Confusion matrix and Classification report
print(confusion_matrix(y_test,predictions))
print("/n")
print(classification_report(y_test,predictions))

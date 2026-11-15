from data import X_train,X_test,y_train,y_test,X,y
import matplotlib.pyplot as plt
import seaborn as sns
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

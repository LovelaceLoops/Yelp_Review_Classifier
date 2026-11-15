import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
global X_train
global y_train

#Load Dataset
print("======== Load Yelp Review Dataset =======/n")
yelp = pd.read_csv("yelp.csv")
print(yelp.head(5))
print(yelp.info())
print(yelp.describe())

#New length column "Text Length" (no. of words in text column)
yelp['text length'] = yelp['text'].apply(len)
print(yelp.head(5))

#Exploratory data analysis
print("======= Exploratory Data Analysis========/n")
sns.set_style('white')

grid = sns.FacetGrid(yelp,col='stars')
grid.map(plt.hist,'text length',bins = 20)
plt.show()

sns.boxplot(x='stars',y='text length',data=yelp)
plt.show()

sns.countplot(x='stars',data=yelp)
plt.show()

group = yelp.groupby('stars')
stars = group.mean(numeric_only=True)
print(stars)

print("===Correlation Matrix===")
corr = stars.corr()
print(corr)
sns.heatmap(corr, annot=True,cmap='coolwarm')
plt.show()

#Prepare data
print("Filter the dataset with only 1 and 5 star review")
yelp_class = yelp[(yelp['stars']==1) | (yelp['stars']==5)]
print(yelp_class.info())

X = yelp_class['text']
y = yelp_class['stars']

cv = CountVectorizer()
X = cv.fit_transform(X)

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

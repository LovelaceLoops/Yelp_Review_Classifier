import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

sns.countplot(x='stars',data=yelp,palette='rainbow')
plt.show()

group = yelp.groupby('stars')
stars = group.mean(numeric_only=True)
print(stars)

print("===Correlation Matrix===")
corr = stars.corr()
print(corr)
sns.heatmap(corr, annot=True,cmap='coolwarm')
plt.show()

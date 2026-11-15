import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load Dataset
print("======== Load Yelp Review Dataset =======/n")
yelp = pd.read_csv("yelp.csv")
# print(yelp.head(5))
# print(yelp.info())
print(yelp.describe())

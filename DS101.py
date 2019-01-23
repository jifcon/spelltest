import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd
import seaborn as sns
sns.set (style="ticks")
from sklearn.feature_extraction.text import CountVectorizer
from skimage.io import imread, imshow

titanic = sns.load_dataset("titanic")
titanic.head()
titanic.embark_town.unique()

pd.get_dummies(titanic,columns=['embark_town']).head()
pd.get_dummies(titanic,columns=['embarked']).head()

titanic_num = pd.get_dummies(titanic,columns=['sex','class','who',])

titanic_fare = pd.DataFrame(titanic.fare.unique())

text = "This is a new text written to text a bag of words in a real world example"
cv = CountVectorizer()
feat = cv.fit_transform([text])
for word,idx in cv.vocabulary_.items ():
    print("%-14s%d" % (word, feat[0,idx]))

print()


#Decision Trees












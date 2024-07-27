import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import numpy as np
path = "d:/projects/Trainning/ML/Assignment/HousingModified.csv"
data = pd.read_csv(path)

print (data.describe())

setx = data[['area','bedrooms','bathrooms']]
sety = data['price']

# get the data
trainx,testx,trainy,testy =train_test_split(setx,sety,test_size=0.1,random_state=42)

# train 
lr  = LinearRegression()
lr.fit(trainx,trainy)
# test 
predict = lr.predict(testx)


plt.scatter(predict,testy)
plt.xlabel("Prediction")
plt.ylabel("TestPrice")
plt.show()




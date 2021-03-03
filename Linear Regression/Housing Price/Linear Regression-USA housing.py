#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#reading the csv file
USAhousing = pd.read_csv('USA_Housing.csv')


#checking the first 5 rows
USAhousing.head()


#getting the info
USAhousing.info()
USAhousing.describe()
USAhousing.columns

#visualizing the dataset
sns.pairplot(USAhousing)
sns.distplot(USAhousing['Price'])
sns.heatmap(USAhousing.corr())


#Training a Linear Regression Model

#Split data into an X array that contains the features to train on, and a y array with the target variable
#Address column deleted as it contains text info that the linear regression model can't use.

X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']


#Train Test Split
#Split the data into training set and a testing set. 
#Train model on the training set and evaluate the model using test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)


#Model Evaluation
#Evaluate the model by checking out it's coefficients

print(lm.intercept_)

coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
print(coeff_df)


#Predictions from Model
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)


#Residual Histogram
sns.distplot((y_test-predictions),bins=50);


#Regression Evaluation Metrics
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


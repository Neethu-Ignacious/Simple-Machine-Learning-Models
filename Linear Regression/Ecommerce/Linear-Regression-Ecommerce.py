#Linear Regression
#Ecommerce Customers

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Get data
customers=pd.read_csv("Ecommerce Customers")
customers.head()
customers.info()
customers.describe()

#visualize data
sns.set_style("whitegrid")
sns.jointplot(data=customers,x="Time on Website",y="Yearly Amount Spent")
sns.jointplot(data=customers,x="Time on App",y="Yearly Amount Spent")
sns.jointplot(data=customers,x="Time on App",y="Length of Membership",kind="hex")
sns.pairplot(customers)
sns.lmplot(data=customers,x="Length of Membership",y="Yearly Amount Spent")

#splitting the data
y=customers['Yearly Amount Spent']
X=customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

#Training the model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.coef_)

#Predicting Test data
predictions=lm.predict(X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

#Evaluating the model

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

sns.distplot((y_test-predictions),bins=50);

coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
print(coeff_df)


#import libraries
import pandas as pd
import seaborn as sns

#get data
ad_data=pd.read_csv("advertising.csv")
ad_data.head()
ad_data.info()
ad_data.describe()

#visualizing data
sns.set_style("whitegrid")
sns.distplot(ad_data["Age"],kde=False,bins=30)
sns.jointplot(data=ad_data,x="Age",y="Area Income",kind="scatter")
sns.jointplot(data=ad_data,x="Age",y="Daily Time Spent on Site",kind="kde",color="red")
sns.jointplot(data=ad_data,x="Daily Time Spent on Site",y="Daily Internet Usage",kind="scatter",color="green")
sns.pairplot(data=ad_data,hue="Clicked on Ad",palette="husl")
sns.heatmap(ad_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#preparing data
ad_data.drop(["Ad Topic Line","City","Country","Timestamp"],axis=1,inplace=True)
ad_data.head()

#Splitting data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(ad_data.drop("Clicked on Ad",axis=1),ad_data["Clicked on Ad"],test_size=0.30,
                                              random_state=101)

#Training the model
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

#Predicting
predictions = logmodel.predict(X_test)
predictions


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

#Testing the data
df_test=pd.read_csv("advertising_test_data.csv")
df_test.drop(["Ad Topic Line","City","Country","Timestamp"],axis=1,inplace=True)
df_test

predictions = logmodel.predict(df_test)
predictions


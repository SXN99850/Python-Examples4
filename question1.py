import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns

#For data visualisation using seaborn sns.set() method
sns.set(style="white", color_codes=True)

import warnings
warnings.filterwarnings("ignore")

#import pandas and read dataset using read_csv method
df=pd.read_csv("./Salary_Data.csv")
print(df.head())
#Slicing the necessary features using iloc
X = df.iloc[:, :-1].values
Y = df.iloc[:, 1].values
#train_test_split the data using the sklearn method train_test_split()
X_Training, X_Testing, Y_Training, Y_Testing = train_test_split(X,Y, test_size=1/3,random_state = 0)

#Using linear regression -train and predict model
#Linear regression uses the relationship between the data-points to draw a straight line through all them.
# This line can be used to predict future values.
regressor = LinearRegression()
regressor.fit(X_Training, Y_Training)
Y_Predict = regressor.predict(X_Testing)

#calculating mean_squared_error using the available method from sklearn metrics
print("Mean squared error", mean_squared_error(Y_Testing,Y_Predict))

#Scatterplot for Training Data
plt.title('Training data')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.scatter(X_Training, Y_Training)
plt.show()
#Scatterplot for Testing Data
plt.title('Testing data')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.scatter(X_Testing, Y_Testing)
plt.show()
#Name: Anooshka Bajaj

import pandas as pd
df=pd.read_csv(r'E:\Sem 3\IC272 Data Science 3\Lab 5\atmosphere_data.csv')
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures  
import numpy as np



#splitting the data into training and test data
[X_train, X_test, X_label_train, X_label_test] =train_test_split(df[df.columns[:-1]], df['temperature'], test_size=0.3, random_state=42,shuffle=True)

training_data = pd.concat((X_train,X_label_train),axis=1)
training_data.to_csv('atmosphere_train.csv',index = False)

test_data = pd.concat((X_test,X_label_test),axis=1)
test_data.to_csv('atmosphere_test.csv',index = False)


#1
regressor = LinearRegression()

x_train=training_data.iloc[:,1].values.reshape(-1,1)         #training data: pressure
y_train=training_data.iloc[:,-1].values.reshape(-1,1)        #training data: temperature
regressor.fit(x_train,y_train)                               #training the algorithm

x_test=test_data.iloc[:,1].values.reshape(-1,1)              #test data: pressure
y_test=test_data.iloc[:,-1].values.reshape(-1,1)             #test data: temperature
y_pred = regressor.predict(x_test)                           #making predictions

#1(a)
plt.scatter(x_train,y_train)
plt.plot(x_train,regressor.predict(x_train),c='red')
plt.xlabel('Pressure')
plt.ylabel('Temperature')
plt.title("Best Fit Line")
plt.show()

#1(b)
print('\nPrediction accuracy for training data:')
print((mean_squared_error(y_train,regressor.predict(x_train)))**0.5)

#1(c)
print('\nPrediction accuracy for test data:')
print((mean_squared_error(y_test,regressor.predict(x_test)))**0.5)

#1(d)
plt.scatter(y_test,y_pred)
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.title('Scatter plot of Actual Temperature v/s Predicted Temperature')
plt.show()


#2
def polynomialcurvefitting (p,train_data,y_train,test_data):
    polynomial_features = PolynomialFeatures(p)                      #degree=p
    train_poly = polynomial_features.fit_transform(train_data)
    test_poly = polynomial_features.fit_transform(test_data)
    regressor = LinearRegression()
    regressor.fit(train_poly,y_train)
    
    pred = regressor.predict(test_poly)
    return pred

#2(a)
RMSE = []                    
minrmse = 10**10             #for storing min RMSE 
minrmse_p = 0                #for storing degree with min RMSE
for p in range(2,6):
    pcf1= polynomialcurvefitting (p,x_train,y_train,x_train)  
    rmse = (mean_squared_error(pcf1,y_train))**0.5
    RMSE.append(rmse)
    if minrmse > RMSE[-1]:
        minrmse = RMSE[-1]
        minrmse_p = p
        
plt.bar(range(2,6),RMSE)
plt.xlabel('Polynomial Degree (p)')
plt.ylabel('RMSE')
plt.title('Prediction Accuracy on Training Data')
plt.show()

#2(b)
RMSE = []                    
minrmse = 10**10             #for storing min RMSE 
minrmse_p = 0                #for storing degree with min RMSE
for p in range(2,6):
    pcf2= polynomialcurvefitting (p,x_train,y_train,x_test)  
    rmse = (mean_squared_error(pcf2,y_test))**0.5
    RMSE.append(rmse)
    if minrmse > RMSE[-1]:
        minrmse = RMSE[-1]
        minrmse_p = p
        
plt.bar(range(2,6),RMSE)
plt.xlabel('Polynomial Degree (p)')
plt.ylabel('RMSE')
plt.title('Prediction Accuracy on Test Data')
plt.show()

#2(c)
b = polynomialcurvefitting(minrmse_p,x_train,y_train,x_test)
p = np.polyfit(test_data.iloc[:,1],b,minrmse_p)
xt = np.linspace(x_train.min(),x_train.max())

plt.scatter(x_train,y_train)
plt.plot(xt,np.polyval(p,xt),c='red')
plt.xlabel('Pressure')
plt.ylabel('Temperature')
plt.title("Best Fit Curve")
plt.show()

#2(d)
plt.scatter(y_test,polynomialcurvefitting(minrmse_p,x_train,y_train,x_test))
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.title('Scatter plot for the best degree of polynomial (p)')
plt.show()


    
   
    




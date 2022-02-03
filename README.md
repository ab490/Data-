# Regression
Regression using Simple Linear Regression and Polynomial Curve Fitting.

I am given with data file atmosphere_data.csv that contains the readings from various sensors installed at 10 locations around Mandi district. These sensors measure the 
different atmospheric factors like temperature, humidity, atmospheric pressure, amount of rain, average light, maximum light and moisture content. The goal of this dataset is to model the atmospheric temperature.

I have written a python program to split the data from atmosphere_data.csv into train data and test data and do the following:

1. Built the simple linear regression (straight-line regression) model to predict temperature given pressure.\
a. Plotted the best fit line on the training data. \
b. Found the prediction accuracy on the training data using root mean squarederror.\
c. Found the prediction accuracy on the test data using root mean squared error.\
d. Plotted the scatter plot of actual temperature vs predicted temperature on the test data.

2. Built the simple nonlinear regression model using polynomial curve fitting to predict temperature given pressure.\
a. Found the prediction accuracy on the training data for the different values of degree of polynomial (p = 2, 3, 4, 5) using root mean squared error (RMSE). Plotted the bar graph of RMSE vs different values of degree of polynomial.\
b. Found the prediction accuracy on the test data for the different values of degree of polynomial (p = 2, 3, 4, 5) using root mean squared error (RMSE). Plotted the bar graph of RMSE (y-axis) vs different values of degree of polynomial (x-axis).\
c. Plotted the best fit curve using best fit model on the training data.\
d. Plotted the scatter plot of actual temperature vs predicted temperature on the test data for the best degree of polynomial. Compared the 2 scatter plots

### Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


### Import Dataset
dataset = pd.read_csv('amd_daily.csv')    # Feel free to use your own dataset to experiment.
# Add Day as index column to serve as independent variable
dataset['Day'] = range(1,len(dataset['Close'])+1)


# Data Preprocessing
# Extracting independent (X) and dependent (y) columns/fields/variables. You can experiment with different independent variables.
X_raw = dataset.iloc[:,7:].values        # Independent variables ('Date')
y_raw = dataset.iloc[:,4:5].values       # Dependent variable ('close')
# Reference for iloc: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html

#print(X)
#print(y)

# NOTE: This time we won't split the dataset into training and test set as we want to utilize whole dataset to train the model.


### Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X_raw)
y = sc_y.fit_transform(y_raw)
# We scale both X & y to implement SVR because the difference between both variables is big enough to create an imbalance of impact on the mathematical formula.
# For eg. In the equation if value of one variable is 5 and other variable is 500.
            # So in such a case, we need to scale both variables in such a way that they get mapped on to same range or range close to each other.

print(X)    # Uncomment these to see how the variables changed after scaling compared to the original values.
print(y)


### Model Training
from sklearn.svm import SVR

regressor = SVR(kernel='rbf')       # Define regressor as SVR instance, but this time with parameter 'kernel' as 'rbf'.
                                    # RBF is one of many techniques called kernel specialized for SVR and SVMs (Support Vector Machines).
regressor.fit(X,y)      # Train the SVR model on the dependent and independent variables/data.


### Predicting Results
#print(sc_y.inverse_transform(regressor.predict(sc_X.transform([[265]])).reshape(1, -1)))

input = [[165]]
input_scaled = sc_X.transform([[165]])      # To predict, we will also have to scale the input as the model is trained on scaled data.
result = regressor.predict(input_scaled)    # Then we get the predicted value, which comes out as already scaled as model trains with scaled target variable. 
result_inversed = sc_y.inverse_transform(result.reshape(1,-1))      # To get actual value, we need to inverse the scaling/transformation.
print(result_inversed)      # Now you have the actual result value.

# To understand the whole flow, check the following:
# Input > Transformation/Scaling > MODEL > Inverse transformation > Actual Result


### If you were able to run till here successfully, then machine learning part is complete. Let's visualize the model.


### Model Visualization

plt.scatter(X_raw,y_raw, color="red")
plt.plot(X_raw, sc_y.inverse_transform(regressor.predict(X).reshape(1, -1))[0], color="blue")
# !Got the following error in the previous line: x and y must have same first dimension, but have shapes (252, 1) and (1, 252)
# So from the error, we can identify that the we need same shape for both X & y. So I reshaped the X.
plt.title("AMD Stock SVR Analysis")
plt.xlabel("Day")
plt.ylabel("Price")
plt.show()

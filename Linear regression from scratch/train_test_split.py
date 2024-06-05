# Importing the necessary libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Splitting the dataset into the Training set, Cross Validation set, and Test set

# Assuming the data is in a DataFrame with more than one column and rows and one target column
df = pd.DataFrame("Your dataframe path here")

# Convert DataFrame to numpy arrays
X = df[['X1', 'X2', ...,"Xn"]].values # Choose the feature columns you want to include
Y = df['Your target column'].values

# Calculate the number of rows in the dataset
n = len(Y)

# Split 60% of the dataset as the training set
train_end = int(0.6 * n)
x_train = X[:train_end]
y_train = Y[:train_end]

# Split the remaining 40% into two: one half for cross validation and the other for the test set
cv_end = int(0.5 * (n - train_end))
x_cv = X[train_end:train_end + cv_end]
y_cv = Y[train_end:train_end + cv_end]
x_test = X[train_end + cv_end:]
y_test = Y[train_end + cv_end:]

print(f"""The shape of the training set (input) is: {x_train.shape}\n
      The shape of the training set (target) is: {y_train.shape}\n
      The shape of the cross validation set (input) is: {x_cv.shape}\n
      The shape of the cross validation set (target) is: {y_cv.shape}\n
      The shape of the test set (input) is: {x_test.shape}\n
      The shape of the test set (target) is: {y_test.shape}""")


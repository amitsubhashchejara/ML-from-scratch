import numpy as np

# Formula to normalize the data is:
# (x - mean) / standard deviation
# This is z-score normalization
# defining a function to normalize the data
def normalize(x):
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

x_train_norm = normalize(x_train)
x_cv_norm = normalize(x_cv)
x_test_norm = normalize(x_test)

print(x_train_norm.shape, x_cv_norm.shape, x_test_norm.shape)